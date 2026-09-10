from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from training.zero_knowledge.encoding import collate_actions, collate_observations
from training.zero_knowledge.env import (
    Action,
    ActionKind,
    DEED_INDEX,
    GROUP_OF,
    GROUPS,
    PRICES,
    RENT_TABLE,
)
from training.zero_knowledge.model import ModelConfig, PropertyTradingPolicy


class TorchCheckpointPolicy:
    def __init__(self, path: Path, device: str):
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu",
        )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model = PropertyTradingPolicy(ModelConfig(**payload["model_config"])).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.hidden_by_seat: dict[int, torch.Tensor] = {}

    def start_game(self, _assignment: object = None) -> None:
        self.hidden_by_seat.clear()

    def __call__(self, view: dict[str, object]) -> Action:
        seat = int(view["seat"])
        actions = view["legal_actions"]
        observation = collate_observations([view["observation"]], self.device)
        action_data, mask = collate_actions(
            [actions],
            [int(view["actor"])],
            [int(view["player_count"])],
            self.device,
        )
        with torch.no_grad():
            logits, _, hidden = self.model(
                observation,
                action_data,
                mask,
                self.hidden_by_seat.get(seat),
            )
        self.hidden_by_seat[seat] = hidden.detach()
        return actions[int(torch.argmax(logits[0]).item())]


class LegacyNeuroPolicy:
    """Behavioral adapter for the frozen neuroevolution-v5 JSON artifact.

    The six output heads have different meanings and thresholds.  They must not
    be compared with one another as if they were action logits.
    """

    BUY = 0
    AUCTION_BID = 1
    BUILD = 2
    LEAVE_JAIL = 3
    CASH_RESERVE = 4
    TRADE = 5
    MAX_HIDDEN = 6

    def __init__(self, path: Path):
        candidates = [path]
        if not path.stem.endswith(("-3p", "-5p")):
            candidates.extend(
                path.with_name(f"{path.stem}-{player_count}p{path.suffix}")
                for player_count in (3, 5)
            )
        self.payloads: dict[int, dict[str, object]] = {}
        for candidate in candidates:
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            player_count = int(payload.get("playerCount", 4))
            self.payloads[player_count] = payload
        if not self.payloads or any(
            payload.get("version") != "league-neuroevolution-v5-marginal-trade-valuation"
            for payload in self.payloads.values()
        ):
            raise ValueError("unsupported legacy neuroevolution policy")

    def start_game(self, _assignment: object = None) -> None:
        return None

    @staticmethod
    def _printed_price(tile: int) -> int:
        return int(PRICES[tile]) if 0 <= tile < len(PRICES) else 0

    def _features(self, view: dict[str, object], action: Action) -> list[float]:
        if action.kind == ActionKind.TRADE:
            return self._trade_proposal_features(view, action)
        observation = view["observation"]
        global_features = np.asarray(observation["global"])
        players = np.asarray(observation["players"])
        properties = np.asarray(observation["properties"])
        cash = float(players[0, 4] * 3000)
        price = float(abs(action.cash) or self._printed_price(action.tile))
        post_cash = cash - max(0.0, float(action.cash))
        rent = float(RENT_TABLE.get(action.tile, (0,))[0])
        group_progress = 0.0
        completes_set = 0.0
        deed_index = DEED_INDEX.get(action.tile)
        if deed_index is not None and action.tile in GROUP_OF:
            group = GROUP_OF[action.tile]
            group_indices = [DEED_INDEX[tile] for tile in GROUPS[group]]
            owned = sum(math.isclose(float(properties[index, 5]), 1 / 6, abs_tol=1e-5) for index in group_indices)
            group_progress = owned / len(group_indices)
            completes_set = float(owned + 1 == len(group_indices))
        living_cash = [float(row[4] * 3000) for row in players if row[0] > 0 and row[1] > 0]
        advantage = cash - (float(np.mean(living_cash[1:])) if len(living_cash) > 1 else cash)
        return [
            float(np.clip(cash / 1500, -1, 2)),
            float(np.clip(post_cash / 1500, -1, 2)),
            float(np.clip(price / 400, 0, 2)),
            float(np.clip(rent / 200, -2, 2)),
            group_progress,
            completes_set,
            float(np.clip(global_features[9], 0, 1)),
            float(np.clip(advantage / 2000, -1, 1)),
            float(np.clip(global_features[28], -1, 1)),
        ]

    def _trade_property_value(
        self,
        properties: np.ndarray,
        tile: int,
        *,
        acquiring: bool,
    ) -> tuple[float, float, float]:
        if tile < 0:
            return 0.0, 0.0, 0.0
        base = float(self._printed_price(tile))
        income = float(RENT_TABLE.get(tile, (0,))[0] * 4)
        if tile not in GROUP_OF:
            return base + income, 0.0, 0.0
        group = GROUP_OF[tile]
        indices = [DEED_INDEX[item] for item in GROUPS[group]]
        count = sum(
            math.isclose(float(properties[index, 5]), 1 / 6, abs_tol=1e-5)
            for index in indices
        )
        size = len(indices)
        after = max(0, min(size, count + (1 if acquiring else -1)))
        group_bonus = lambda owned: base * 0.35 * (owned / size) + (base * 0.9 if owned == size else 0.0)
        marginal = (
            group_bonus(after) - group_bonus(count)
            if acquiring
            else group_bonus(count) - group_bonus(after)
        )
        progress = after / size
        completes = float(acquiring and after == size)
        return base + income + marginal, progress, completes

    def _trade_proposal_features(
        self,
        view: dict[str, object],
        action: Action,
    ) -> list[float]:
        observation = view["observation"]
        global_features = np.asarray(observation["global"])
        players = np.asarray(observation["players"])
        properties = np.asarray(observation["properties"])
        cash = self._cash(view)
        received, progress, completes = self._trade_property_value(
            properties,
            action.take_tile,
            acquiring=True,
        )
        surrendered, give_progress, _ = self._trade_property_value(
            properties,
            action.give_tile,
            acquiring=False,
        )
        if action.take_tile < 0:
            progress = give_progress
        received += action.take_card * 50
        surrendered += action.give_card * 50
        outcome = received - surrendered - action.cash
        price = max(abs(action.cash), received, surrendered)
        other_cash = [float(row[4] * 3000) for row in players[1:] if row[0] > 0 and row[1] > 0]
        advantage = cash - (float(np.mean(other_cash)) if other_cash else cash)
        return [
            float(np.clip(cash / 1500, -1, 2)),
            float(np.clip((cash - action.cash) / 1500, -1, 2)),
            float(np.clip(price / 400, 0, 2)),
            float(np.clip(outcome / 200, -2, 2)),
            float(np.clip(progress, 0, 1)),
            completes,
            float(np.clip(global_features[9], 0, 1)),
            float(np.clip(advantage / 2000, -1, 1)),
            float(np.clip(global_features[33], 0, 1)),
        ]

    def _network_score(
        self,
        payload: dict[str, object],
        features: list[float],
        output: int,
    ) -> float:
        hidden_count = int(payload["hiddenCount"])
        input_count = int(payload["inputCount"])
        value = float(payload["outputBias"][output])
        for hidden in range(hidden_count):
            offset = hidden * input_count
            total = float(payload["hiddenBias"][hidden])
            total += sum(
                float(payload["inputHidden"][offset + index]) * feature
                for index, feature in enumerate(features)
            )
            value += math.tanh(total) * float(
                payload["hiddenOutput"][output * self.MAX_HIDDEN + hidden],
            )
        return math.tanh(value)

    def _cash_reserve(self, payload: dict[str, object], features: list[float]) -> int:
        score = self._network_score(payload, features, self.CASH_RESERVE)
        return 75 + math.floor(((score + 1) * 0.5) * 500)

    def _auction_ceiling(
        self,
        payload: dict[str, object],
        features: list[float],
        *,
        cash: int,
        price: int,
        completes_set: bool,
    ) -> int:
        reserve = self._cash_reserve(payload, features)
        value = (self._network_score(payload, features, self.AUCTION_BID) + 1) * 0.5
        multiplier = 0.28 + value * 1.35 + (0.32 if completes_set else 0.0)
        return max(0, min(cash - reserve, math.floor(price * multiplier)))

    @staticmethod
    def _cash(view: dict[str, object]) -> int:
        return round(float(np.asarray(view["observation"]["players"])[0, 4]) * 3000)

    def _choose_purchase(
        self,
        payload: dict[str, object],
        view: dict[str, object],
        buy: Action,
        decline: Action,
    ) -> Action:
        cash = self._cash(view)
        features = self._features(view, buy)
        reserve = self._cash_reserve(payload, features)
        completes = bool(features[5])
        ceiling = self._auction_ceiling(
            payload,
            features,
            cash=cash,
            price=PRICES[buy.tile],
            completes_set=completes,
        )
        willing = (
            (
                self._network_score(payload, features, self.BUY) > -0.05
                or ceiling >= PRICES[buy.tile]
            )
            and cash - PRICES[buy.tile] >= reserve * 0.38
        )
        return buy if willing else decline

    def _choose_auction(
        self,
        payload: dict[str, object],
        view: dict[str, object],
        bids: list[Action],
        passed: Action,
    ) -> Action:
        if not bids:
            return passed
        tile = bids[0].tile
        reference = Action(ActionKind.BUY, tile=tile, cash=PRICES[tile])
        features = self._features(view, reference)
        ceiling = self._auction_ceiling(
            payload,
            features,
            cash=self._cash(view),
            price=PRICES[tile],
            completes_set=bool(features[5]),
        )
        affordable = [action for action in bids if action.cash <= ceiling]
        return max(affordable, key=lambda action: action.cash) if affordable else passed

    def _trade_response_features(self, view: dict[str, object]) -> list[float]:
        observation = view["observation"]
        global_features = np.asarray(observation["global"])
        properties = np.asarray(observation["properties"])
        cash = self._cash(view)
        offered_cash = round(float(global_features[28]) * 1500)
        give_tile = round(float(global_features[29]) * 39) if global_features[29] >= 0 else -1
        take_tile = round(float(global_features[30]) * 39) if global_features[30] >= 0 else -1
        # From the responder's perspective, the proposer gives give_tile/cash
        # and receives take_tile.  Reuse the same marginal group valuation as
        # proposal scoring so breaking a completed group remains visible.
        received, receive_progress, _ = self._trade_property_value(
            properties,
            give_tile,
            acquiring=True,
        )
        surrendered, surrender_progress, _ = self._trade_property_value(
            properties,
            take_tile,
            acquiring=False,
        )
        received += round(float(global_features[31])) * 50
        surrendered += round(float(global_features[32])) * 50
        outcome = received - surrendered + offered_cash
        price = abs(offered_cash) or max(received, surrendered)
        progress = surrender_progress if take_tile >= 0 else receive_progress
        living = np.asarray(observation["players"])
        other_cash = [float(row[4] * 3000) for row in living[1:] if row[0] > 0 and row[1] > 0]
        advantage = cash - (float(np.mean(other_cash)) if other_cash else cash)
        return [
            float(np.clip(cash / 1500, -1, 2)),
            float(np.clip((cash + offered_cash) / 1500, -1, 2)),
            float(np.clip(price / 400, 0, 2)),
            float(np.clip(outcome / 200, -2, 2)),
            float(np.clip(progress, 0, 1)),
            0.0,
            float(np.clip(global_features[9], 0, 1)),
            float(np.clip(advantage / 2000, -1, 1)),
            float(np.clip(global_features[33], 0, 1)),
        ]

    def __call__(self, view: dict[str, object]) -> Action:
        player_count = int(view["player_count"])
        payload = self.payloads.get(player_count)
        if payload is None:
            raise ValueError(f"legacy checkpoint has no {player_count}-player policy")
        actions = list(view["legal_actions"])
        by_kind = {
            kind: [action for action in actions if action.kind == kind]
            for kind in ActionKind
        }

        if by_kind[ActionKind.BUY]:
            return self._choose_purchase(
                payload,
                view,
                by_kind[ActionKind.BUY][0],
                by_kind[ActionKind.DECLINE][0],
            )
        if by_kind[ActionKind.PASS_AUCTION]:
            return self._choose_auction(
                payload,
                view,
                by_kind[ActionKind.BID],
                by_kind[ActionKind.PASS_AUCTION][0],
            )
        if by_kind[ActionKind.ACCEPT_TRADE]:
            features = self._trade_response_features(view)
            return (
                by_kind[ActionKind.ACCEPT_TRADE][0]
                if self._network_score(payload, features, self.TRADE) > -0.08
                else by_kind[ActionKind.REJECT_TRADE][0]
            )

        cash = self._cash(view)
        builds = []
        for action in by_kind[ActionKind.BUILD]:
            features = self._features(view, action)
            if (
                cash - action.cash >= self._cash_reserve(payload, features)
                and self._network_score(payload, features, self.BUILD) > -0.08
            ):
                builds.append((self._network_score(payload, features, self.BUILD), action))
        if builds:
            return max(builds, key=lambda item: item[0])[1]

        trades = []
        for action in by_kind[ActionKind.TRADE]:
            features = self._features(view, action)
            if (
                cash - max(0, action.cash) >= self._cash_reserve(payload, features) * 0.45
                and self._network_score(payload, features, self.TRADE) > -0.08
            ):
                trades.append((features[3], action))
        if trades:
            return max(trades, key=lambda item: item[0])[1]

        jail_actions = (
            by_kind[ActionKind.USE_JAIL_CARD] + by_kind[ActionKind.PAY_JAIL]
        )
        if jail_actions:
            features = self._features(view, jail_actions[0])
            if self._network_score(payload, features, self.LEAVE_JAIL) > 0:
                return jail_actions[0]
        if by_kind[ActionKind.ROLL_JAIL]:
            return by_kind[ActionKind.ROLL_JAIL][0]

        global_features = np.asarray(view["observation"]["global"])
        in_debt = bool(global_features[1 + 5])
        if in_debt:
            liquidations = by_kind[ActionKind.SELL_BUILDING] + by_kind[ActionKind.MORTGAGE]
            if liquidations:
                return max(liquidations, key=lambda action: action.cash)
            if by_kind[ActionKind.BANKRUPT]:
                return by_kind[ActionKind.BANKRUPT][0]

        for kind in (ActionKind.UNMORTGAGE, ActionKind.ROLL, ActionKind.END_TURN):
            if by_kind[kind]:
                return by_kind[kind][0]
        for kind in (ActionKind.MORTGAGE, ActionKind.SELL_BUILDING, ActionKind.BANKRUPT):
            if by_kind[kind]:
                return by_kind[kind][0]
        raise RuntimeError("legacy policy received no supported legal action")


def load_frozen_policy(path: Path, *, device: str):
    if path.suffix.lower() == ".json":
        return LegacyNeuroPolicy(path)
    return TorchCheckpointPolicy(path, device)
