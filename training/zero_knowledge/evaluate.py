from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .env import Action, ActionKind, PRICES, PropertyTradingEnv
from .model import ModelConfig, PropertyTradingPolicy, collate_actions, collate_observations


def baseline_action(env: PropertyTradingEnv, actions: list[Action], style: str) -> int:
    reserve = {"conservative": 500, "balanced": 300, "aggressive": 120}[style]
    actor = env.actor
    by_kind: dict[ActionKind, list[tuple[int, Action]]] = {}
    for index, action in enumerate(actions):
        by_kind.setdefault(action.kind, []).append((index, action))

    if ActionKind.ACCEPT_TRADE in by_kind:
        trade = env.pending_trade
        if trade is None:
            return by_kind[ActionKind.REJECT_TRADE][0][0]
        received = max(0, trade.cash) + (int(PRICES[trade.give_tile]) if trade.give_tile >= 0 else 0)
        given = max(0, -trade.cash) + (int(PRICES[trade.take_tile]) if trade.take_tile >= 0 else 0)
        kind = ActionKind.ACCEPT_TRADE if received >= given else ActionKind.REJECT_TRADE
        return by_kind[kind][0][0]
    if ActionKind.BUY in by_kind:
        index, action = by_kind[ActionKind.BUY][0]
        if env.cash[actor] - action.cash >= reserve:
            return index
        return by_kind[ActionKind.DECLINE][0][0]
    if ActionKind.BID in by_kind:
        tile = actions[0].tile
        multiplier = {"conservative": 0.7, "balanced": 1.0, "aggressive": 1.25}[style]
        ceiling = min(int(env.cash[actor] - reserve * 0.5), int(PRICES[tile] * multiplier))
        choices = [(index, action) for index, action in by_kind[ActionKind.BID] if action.cash <= ceiling]
        if choices:
            return max(choices, key=lambda item: item[1].cash)[0]
        return by_kind[ActionKind.PASS_AUCTION][0][0]
    if ActionKind.BANKRUPT in by_kind:
        cash_raising = by_kind.get(ActionKind.SELL_BUILDING, []) or by_kind.get(ActionKind.MORTGAGE, [])
        return cash_raising[0][0] if cash_raising else by_kind[ActionKind.BANKRUPT][0][0]
    if ActionKind.BUILD in by_kind and env.cash[actor] >= reserve + 200:
        return by_kind[ActionKind.BUILD][0][0]
    if ActionKind.UNMORTGAGE in by_kind and env.cash[actor] >= reserve + 300:
        return by_kind[ActionKind.UNMORTGAGE][0][0]
    if ActionKind.PAY_JAIL in by_kind and env.cash[actor] >= reserve + 250:
        return by_kind[ActionKind.PAY_JAIL][0][0]
    if ActionKind.USE_JAIL_CARD in by_kind:
        return by_kind[ActionKind.USE_JAIL_CARD][0][0]
    for kind in (ActionKind.ROLL_JAIL, ActionKind.ROLL, ActionKind.END_TURN, ActionKind.DECLINE, ActionKind.REJECT_TRADE):
        if kind in by_kind:
            return by_kind[kind][0][0]
    return 0


def load_model(path: Path, device: torch.device) -> PropertyTradingPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PropertyTradingPolicy(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def model_action(model: PropertyTradingPolicy, env: PropertyTradingEnv, actions: list[Action], hidden: torch.Tensor, device: torch.device) -> tuple[int, torch.Tensor]:
    observation = collate_observations([env.observe()], device)
    action_data, mask = collate_actions([actions], [env.actor], [env.player_count], device)
    with torch.no_grad():
        logits, _, next_hidden = model(observation, action_data, mask, hidden)
    return int(Categorical(logits=logits.float()).sample().item()), next_hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--style", choices=("conservative", "balanced", "aggressive"), default="balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(Path(args.checkpoint), device)
    rng = np.random.default_rng(args.seed)
    results: dict[str, object] = {}
    for player_count in [int(value) for value in args.players.split(",")]:
        wins = 0
        ranks: list[int] = []
        for game in range(args.games):
            env = PropertyTradingEnv(player_count, seed=int(rng.integers(2**31 - 1)), max_rounds=360)
            model_seat = game % player_count
            hidden = torch.zeros(1, model.hidden_size, device=device)
            while not env.done:
                actions = env.legal_actions()
                if env.actor == model_seat:
                    index, hidden = model_action(model, env, actions, hidden, device)
                else:
                    index = baseline_action(env, actions, args.style)
                env.step(actions[index], validate=False)
            reward = float(env.terminal_rewards[model_seat])
            rank = int(round(1 + (1 - reward) * (player_count - 1) / 2))
            ranks.append(rank)
            wins += int(rank == 1)
        results[str(player_count)] = {
            "games": args.games,
            "wins": wins,
            "win_rate": wins / args.games,
            "average_rank": float(np.mean(ranks)),
            "baseline": args.style,
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
