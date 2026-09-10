from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

from .board_rules import indexed_board_rules
from .encoding import (
    MAX_PLAYERS,
    encode_action_features,
    encode_observation,
)

_BOARD_RULES = indexed_board_rules()
BOARD_SIZE = _BOARD_RULES["board_size"]
MAX_ROUNDS = 360

DEED_TILES = _BOARD_RULES["deed_tiles"]
DEED_INDEX = {tile: index for index, tile in enumerate(DEED_TILES)}

PRICES = _BOARD_RULES["prices"]
MORTGAGES = _BOARD_RULES["mortgages"]
BUILD_COST = _BOARD_RULES["build_costs"]
RENT_TABLE = _BOARD_RULES["rent_table"]
GROUPS = {
    index: _BOARD_RULES["groups"][group]
    for index, group in enumerate(_BOARD_RULES["group_order"][:8])
}
GROUP_OF = {tile: group for group, tiles in GROUPS.items() for tile in tiles}
RAILROADS = _BOARD_RULES["groups"]["station"]
UTILITIES = _BOARD_RULES["groups"]["utility"]
CHANCE_TILES = (7, 22, 36)
COMMUNITY_TILES = (2, 17, 33)

PHASES = (
    "turn_start", "jail", "buy", "auction", "post_turn", "debt", "trade_response", "done"
)


class ActionKind(IntEnum):
    ROLL = 0
    END_TURN = 1
    BUY = 2
    DECLINE = 3
    BID = 4
    PASS_AUCTION = 5
    PAY_JAIL = 6
    USE_JAIL_CARD = 7
    ROLL_JAIL = 8
    MORTGAGE = 9
    UNMORTGAGE = 10
    BUILD = 11
    SELL_BUILDING = 12
    TRADE = 13
    ACCEPT_TRADE = 14
    REJECT_TRADE = 15
    BANKRUPT = 16


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    tile: int = -1
    target: int = -1
    cash: int = 0
    give_tile: int = -1
    take_tile: int = -1
    give_card: int = 0
    take_card: int = 0

    def key(self) -> tuple[int, ...]:
        return (
            int(self.kind), self.tile, self.target, self.cash,
            self.give_tile, self.take_tile, self.give_card, self.take_card,
        )


class MonopolyEnv:
    """Turn-based classic Monopoly environment with no hand-authored policy values.

    The simulator contains only game rules and printed deed values. All strategic
    preferences are learned from terminal game outcomes.
    """

    def __init__(self, player_count: int = 4, seed: int | None = None, max_rounds: int = MAX_ROUNDS):
        if player_count not in (3, 4, 5):
            raise ValueError("player_count must be 3, 4, or 5")
        self.player_count = player_count
        self.max_rounds = max_rounds
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], list[Action]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.player_count
        self.cash = np.full(n, 1500, dtype=np.int32)
        self.position = np.zeros(n, dtype=np.int8)
        self.alive = np.ones(n, dtype=np.bool_)
        self.in_jail = np.zeros(n, dtype=np.bool_)
        self.jail_turns = np.zeros(n, dtype=np.int8)
        self.jail_cards = np.zeros(n, dtype=np.int8)
        self.jail_card_decks: list[list[bool]] = [[] for _ in range(n)]
        self.owner = np.full(BOARD_SIZE, -1, dtype=np.int8)
        self.houses = np.zeros(BOARD_SIZE, dtype=np.int8)
        self.mortgaged = np.zeros(BOARD_SIZE, dtype=np.bool_)
        self.bank_houses = 32
        self.bank_hotels = 12
        self.active = 0
        self.actor = 0
        self.round = 1
        self.phase = "turn_start"
        self.last_roll = 0
        self.last_doubles = False
        self.doubles_count = 0
        self.pending_tile = -1
        self.debt_creditor = -1
        self.resume_landing_after_debt = False
        self.auction: dict[str, object] | None = None
        self.auction_queue: list[int] = []
        self.auction_return = "after_landing"
        self.pending_trade: Action | None = None
        self.payment_queue: list[tuple[int, int, int]] = []
        self.trade_return_phase = "turn_start"
        self.trades_left = 6
        self.traded_tiles: set[int] = set()
        self.managed_tiles: set[int] = set()
        self.built_tiles: set[int] = set()
        self.sold_tiles: set[int] = set()
        self.rejected_offers: set[tuple[int, ...]] = set()
        self.turn_count = 0
        self.eliminated_order: list[int] = []
        self.done = False
        self.terminal_rewards = np.zeros(n, dtype=np.float32)
        self.chance_deck = list(range(16))
        self.community_deck = list(range(16))
        self.rng.shuffle(self.chance_deck)
        self.rng.shuffle(self.community_deck)
        self.chance_cursor = 0
        self.community_cursor = 0
        return self.observe(), self.legal_actions()

    def clone_seed(self) -> int:
        return int(self.rng.integers(1, 2**31 - 1))

    def validate_state(self) -> None:
        if self.done:
            assert self.phase == "done"
        else:
            assert 0 <= self.actor < self.player_count and self.alive[self.actor]
            assert 0 <= self.active < self.player_count
        assert self.bank_houses >= 0 and self.bank_hotels >= 0
        for tile in range(BOARD_SIZE):
            if tile not in DEED_INDEX:
                assert int(self.owner[tile]) == -1
                assert int(self.houses[tile]) == 0
                assert not self.mortgaged[tile]
                continue
            assert -1 <= int(self.owner[tile]) < self.player_count
            assert 0 <= int(self.houses[tile]) <= 5
            if self.mortgaged[tile]:
                assert int(self.houses[tile]) == 0
                assert not self._group_has_buildings(tile)
            if int(self.houses[tile]) > 0:
                assert tile in GROUP_OF
                assert self._owns_group(int(self.owner[tile]), tile)

    def legal_actions(self) -> list[Action]:
        if self.done:
            return []
        if self.phase == "trade_response":
            return [Action(ActionKind.ACCEPT_TRADE), Action(ActionKind.REJECT_TRADE)]
        if self.phase == "auction":
            return self._auction_actions()

        actions: list[Action] = []
        if self.phase in ("turn_start", "post_turn", "jail", "debt"):
            management = self._management_actions(self.actor)
            if self.phase == "debt":
                management = [
                    action
                    for action in management
                    if action.kind in (ActionKind.MORTGAGE, ActionKind.SELL_BUILDING)
                ]
            actions.extend(management)
            if self.actor == self.active and self.trades_left > 0:
                actions.extend(self._trade_actions(self.actor))

        if self.phase == "turn_start":
            actions.append(Action(ActionKind.ROLL))
        elif self.phase == "jail":
            actions.append(Action(ActionKind.ROLL_JAIL))
            if self.cash[self.actor] >= 50:
                actions.append(Action(ActionKind.PAY_JAIL))
            if self.jail_cards[self.actor] > 0:
                actions.append(Action(ActionKind.USE_JAIL_CARD))
        elif self.phase == "buy":
            tile = self.pending_tile
            if self.cash[self.actor] >= PRICES[tile]:
                actions.append(Action(ActionKind.BUY, tile=tile, cash=int(PRICES[tile])))
            actions.append(Action(ActionKind.DECLINE, tile=tile))
        elif self.phase == "post_turn":
            actions.append(Action(ActionKind.END_TURN))
        elif self.phase == "debt":
            if not any(
                action.kind in (ActionKind.MORTGAGE, ActionKind.SELL_BUILDING, ActionKind.TRADE)
                for action in actions
            ):
                actions.append(Action(ActionKind.BANKRUPT))

        if not actions:
            actions.append(Action(ActionKind.BANKRUPT))
        return self._deduplicate(actions)

    def step(
        self, action: Action, *, validate: bool = True
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        if self.done:
            raise RuntimeError("cannot step a finished game")
        if validate:
            legal = {candidate.key(): candidate for candidate in self.legal_actions()}
            if action.key() not in legal:
                raise ValueError(f"illegal action {action} in phase {self.phase}")
            action = legal[action.key()]
        actor_before = self.actor
        self._apply(action)
        reward = float(self.terminal_rewards[actor_before]) if self.done else 0.0
        info = {
            "actor": actor_before,
            "next_actor": self.actor,
            "phase": self.phase,
            "terminal_rewards": self.terminal_rewards.copy() if self.done else None,
        }
        return self.observe(), reward, self.done, info

    def _apply(self, action: Action) -> None:
        kind = action.kind
        if kind == ActionKind.ROLL:
            self._roll_and_move(self.actor, jail_roll=False)
        elif kind == ActionKind.END_TURN:
            if self.last_doubles and not self.in_jail[self.active] and self.doubles_count < 3:
                self.last_doubles = False
                self.actor = self.active
                self.phase = "turn_start"
            else:
                self._advance_turn()
        elif kind == ActionKind.BUY:
            self.cash[self.actor] -= int(PRICES[action.tile])
            self.owner[action.tile] = self.actor
            self.pending_tile = -1
            self._after_landing()
        elif kind == ActionKind.DECLINE:
            self._start_auction(action.tile)
        elif kind == ActionKind.BID:
            self._auction_bid(action.cash)
        elif kind == ActionKind.PASS_AUCTION:
            self._auction_pass()
        elif kind == ActionKind.PAY_JAIL:
            self.cash[self.actor] -= 50
            self._leave_jail(self.actor)
            self._roll_and_move(self.actor, jail_roll=False)
        elif kind == ActionKind.USE_JAIL_CARD:
            self._return_jail_card(self.actor)
            self._leave_jail(self.actor)
            self._roll_and_move(self.actor, jail_roll=False)
        elif kind == ActionKind.ROLL_JAIL:
            self._roll_and_move(self.actor, jail_roll=True)
        elif kind == ActionKind.MORTGAGE:
            self.mortgaged[action.tile] = True
            self.cash[self.actor] += int(MORTGAGES[action.tile])
            self.managed_tiles.add(action.tile)
            self._after_management()
        elif kind == ActionKind.UNMORTGAGE:
            cost = int(np.ceil(MORTGAGES[action.tile] * 1.1))
            self.cash[self.actor] -= cost
            self.mortgaged[action.tile] = False
            self.managed_tiles.add(action.tile)
            self._after_management()
        elif kind == ActionKind.BUILD:
            self._build(action.tile)
            self.built_tiles.add(action.tile)
            self._after_management()
        elif kind == ActionKind.SELL_BUILDING:
            self._sell_building(action.tile)
            self.sold_tiles.add(action.tile)
            self._after_management()
        elif kind == ActionKind.TRADE:
            self.pending_trade = action
            self.trade_return_phase = self.phase
            self.actor = action.target
            self.phase = "trade_response"
        elif kind == ActionKind.ACCEPT_TRADE:
            self._resolve_trade(True)
        elif kind == ActionKind.REJECT_TRADE:
            self._resolve_trade(False)
        elif kind == ActionKind.BANKRUPT:
            self._bankrupt(self.actor, self.debt_creditor)
        else:
            raise AssertionError(f"unhandled action {kind}")

    def _roll_and_move(self, player: int, jail_roll: bool) -> None:
        die1 = int(self.rng.integers(1, 7))
        die2 = int(self.rng.integers(1, 7))
        total = die1 + die2
        doubles = die1 == die2
        self.last_roll = total
        self.last_doubles = doubles

        if jail_roll:
            if doubles:
                self._leave_jail(player)
                self._move(player, total)
                self.doubles_count = 0
                self.last_doubles = False
                self._resolve_landing(player, no_extra_turn=True)
                return
            self.jail_turns[player] += 1
            if self.jail_turns[player] >= 3:
                self.cash[player] -= 50
                self._leave_jail(player)
                self._move(player, total)
                self.doubles_count = 0
                if self.cash[player] < 0:
                    self.resume_landing_after_debt = True
                    self._enter_debt(player, -1)
                else:
                    self._resolve_landing(player, no_extra_turn=True)
            else:
                self.last_doubles = False
                self.phase = "post_turn"
                self.actor = self.active
            return

        self.doubles_count = self.doubles_count + 1 if doubles else 0
        if self.doubles_count >= 3:
            self._send_to_jail(player)
            self.phase = "post_turn"
            self.actor = self.active
            return
        self._move(player, total)
        self._resolve_landing(player, no_extra_turn=False)

    def _move(self, player: int, spaces: int) -> None:
        old = int(self.position[player])
        new = (old + spaces) % BOARD_SIZE
        if spaces > 0 and old + spaces >= BOARD_SIZE:
            self.cash[player] += 200
        self.position[player] = new

    def _move_to(self, player: int, destination: int, collect_go: bool) -> None:
        old = int(self.position[player])
        if collect_go and destination < old:
            self.cash[player] += 200
        self.position[player] = destination

    def _resolve_landing(self, player: int, no_extra_turn: bool, rent_multiplier: int = 1, utility_special: bool = False) -> None:
        tile = int(self.position[player])
        self.pending_tile = -1
        if tile in DEED_INDEX:
            owner = int(self.owner[tile])
            if owner < 0:
                self.pending_tile = tile
                self.phase = "buy"
                self.actor = player
                return
            if owner != player and self.alive[owner] and not self.mortgaged[tile]:
                amount = self._rent(tile, self.last_roll, utility_special) * rent_multiplier
                self.cash[player] -= amount
                self.cash[owner] += amount
                if self.cash[player] < 0:
                    self._enter_debt(player, owner)
                    return
        elif tile == 4:
            self.cash[player] -= 200
        elif tile == 38:
            self.cash[player] -= 100
        elif tile == 30:
            self._send_to_jail(player)
        elif tile in CHANCE_TILES:
            if self._draw_card(player, chance=True):
                return
        elif tile in COMMUNITY_TILES:
            if self._draw_card(player, chance=False):
                return

        if self.cash[player] < 0:
            self._enter_debt(player, -1)
            return
        if no_extra_turn:
            self.doubles_count = 0
        self._after_landing()

    def _after_landing(self) -> None:
        if self.done:
            return
        self.phase = "post_turn"
        self.actor = self.active

    def _rent(self, tile: int, dice_total: int, utility_special: bool = False) -> int:
        owner = int(self.owner[tile])
        if tile in RAILROADS:
            count = sum(int(self.owner[item]) == owner for item in RAILROADS)
            return (25, 50, 100, 200)[max(0, count - 1)]
        if tile in UTILITIES:
            count = sum(int(self.owner[item]) == owner for item in UTILITIES)
            return dice_total * (10 if utility_special or count == 2 else 4)
        level = int(self.houses[tile])
        rent = RENT_TABLE[tile][level]
        if level == 0 and self._owns_group(owner, tile):
            rent *= 2
        return rent

    def _draw_card(self, player: int, chance: bool) -> bool:
        deck = self.chance_deck if chance else self.community_deck
        card = deck.pop(0)
        jail_card = card == (8 if chance else 4)
        if jail_card:
            self.jail_card_decks[player].append(chance)
        else:
            deck.append(card)
        return self._apply_card(player, chance, card)

    def _apply_card(self, player: int, chance: bool, card: int) -> bool:
        if chance:
            if card == 0:
                self._move_to(player, 39, False)
                self._resolve_landing(player, False)
                return True
            if card == 1:
                self._move_to(player, 0, True)
            elif card == 2:
                self._move_to(player, 24, True)
                self._resolve_landing(player, False)
                return True
            elif card == 3:
                self._move_to(player, 11, True)
                self._resolve_landing(player, False)
                return True
            elif card in (4, 5):
                rail = next(tile for tile in RAILROADS * 2 if tile > int(self.position[player])) if int(self.position[player]) < 35 else 5
                self._move_to(player, rail, True)
                self._resolve_landing(player, False, rent_multiplier=2)
                return True
            elif card == 6:
                utility = 12 if int(self.position[player]) < 12 or int(self.position[player]) >= 28 else 28
                self._move_to(player, utility, True)
                self.last_roll = int(self.rng.integers(1, 7) + self.rng.integers(1, 7))
                self._resolve_landing(player, False, utility_special=True)
                return True
            elif card == 7:
                self.cash[player] += 50
            elif card == 8:
                self.jail_cards[player] += 1
            elif card == 9:
                self.position[player] = (int(self.position[player]) - 3) % BOARD_SIZE
                self._resolve_landing(player, False)
                return True
            elif card == 10:
                self._send_to_jail(player)
            elif card == 11:
                self.cash[player] -= self._repair_bill(player, 25, 100)
            elif card == 12:
                self.cash[player] -= 15
            elif card == 13:
                self._move_to(player, 5, True)
                self._resolve_landing(player, False)
                return True
            elif card == 14:
                if self._pay_each(player, 50):
                    return True
            elif card == 15:
                self.cash[player] += 150
        else:
            if card == 0:
                self._move_to(player, 0, True)
            elif card == 1:
                self.cash[player] += 200
            elif card == 2:
                self.cash[player] -= 50
            elif card == 3:
                self.cash[player] += 50
            elif card == 4:
                self.jail_cards[player] += 1
            elif card == 5:
                self._send_to_jail(player)
            elif card == 6:
                self.cash[player] += 100
            elif card == 7:
                self.cash[player] += 20
            elif card == 8:
                if self._collect_each(player, 10):
                    return True
            elif card == 9:
                self.cash[player] += 100
            elif card == 10:
                self.cash[player] -= 100
            elif card == 11:
                self.cash[player] -= 50
            elif card == 12:
                self.cash[player] += 25
            elif card == 13:
                self.cash[player] -= self._repair_bill(player, 40, 115)
            elif card == 14:
                self.cash[player] += 10
            elif card == 15:
                self.cash[player] += 100
        if self.cash[player] < 0:
            self._enter_debt(player, -1)
            return True
        return False

    def _return_jail_card(self, player: int) -> None:
        chance = self.jail_card_decks[player].pop(0)
        self.jail_cards[player] -= 1
        deck = self.chance_deck if chance else self.community_deck
        deck.append(8 if chance else 4)

    def _repair_bill(self, player: int, per_house: int, per_hotel: int) -> int:
        bill = 0
        for tile in DEED_TILES:
            if int(self.owner[tile]) != player:
                continue
            level = int(self.houses[tile])
            bill += per_hotel if level == 5 else level * per_house
        return bill

    def _pay_each(self, player: int, amount: int) -> bool:
        payments = [(player, other, amount) for other in range(self.player_count) if other != player and self.alive[other]]
        return self._start_payments(payments)

    def _collect_each(self, player: int, amount: int) -> bool:
        payments = [(other, player, amount) for other in range(self.player_count) if other != player and self.alive[other]]
        return self._start_payments(payments)

    def _start_payments(self, payments: list[tuple[int, int, int]]) -> bool:
        self.payment_queue.extend(payments)
        return self._continue_payments()

    def _continue_payments(self) -> bool:
        while self.payment_queue:
            debtor, creditor, amount = self.payment_queue.pop(0)
            if not self.alive[debtor] or not self.alive[creditor]:
                continue
            self.cash[debtor] -= amount
            self.cash[creditor] += amount
            if self.cash[debtor] < 0:
                self._enter_debt(debtor, creditor)
                return True
        self.debt_creditor = -1
        self.actor = self.active
        self.phase = "post_turn"
        return False

    def _start_auction(self, tile: int, return_phase: str = "after_landing") -> None:
        bidders = [player for player in range(self.player_count) if self.alive[player] and self.cash[player] > 0]
        self.auction = {"tile": tile, "bid": 0, "leader": -1, "active": bidders, "cursor": 0, "steps": 0}
        self.auction_return = return_phase
        self.pending_tile = tile
        self.phase = "auction"
        self.actor = bidders[0] if bidders else self.active
        if not bidders:
            self._finish_auction()

    def _auction_actions(self) -> list[Action]:
        assert self.auction is not None
        bid = int(self.auction["bid"])
        player = self.actor
        actions = [Action(ActionKind.PASS_AUCTION, tile=int(self.auction["tile"]))]
        minimum = bid + 10
        if self.cash[player] >= minimum:
            values = {minimum, bid + 20, bid + 50, bid + 100, int(self.cash[player])}
            for value in sorted(value for value in values if minimum <= value <= int(self.cash[player])):
                actions.append(Action(ActionKind.BID, tile=int(self.auction["tile"]), cash=value))
        return actions

    def _auction_bid(self, amount: int) -> None:
        assert self.auction is not None
        self.auction["bid"] = amount
        self.auction["leader"] = self.actor
        self.auction["steps"] = int(self.auction["steps"]) + 1
        self._next_auction_actor()

    def _auction_pass(self) -> None:
        assert self.auction is not None
        active = list(self.auction["active"])
        if self.actor in active:
            active.remove(self.actor)
        self.auction["active"] = active
        self.auction["steps"] = int(self.auction["steps"]) + 1
        self._next_auction_actor()

    def _next_auction_actor(self) -> None:
        assert self.auction is not None
        active = list(self.auction["active"])
        leader = int(self.auction["leader"])
        if len(active) == 0 or (len(active) == 1 and active[0] == leader):
            self._finish_auction()
            return
        cursor = int(self.auction["cursor"])
        for _ in range(self.player_count + 1):
            cursor = (cursor + 1) % self.player_count
            if cursor in active and cursor != leader:
                self.auction["cursor"] = cursor
                self.actor = cursor
                return
        self._finish_auction()

    def _finish_auction(self) -> None:
        if self.auction is not None:
            leader = int(self.auction["leader"])
            bid = int(self.auction["bid"])
            tile = int(self.auction["tile"])
            if leader >= 0 and bid > 0 and self.cash[leader] >= bid:
                self.cash[leader] -= bid
                self.owner[tile] = leader
        self.auction = None
        self.pending_tile = -1
        if self.auction_queue:
            next_tile = self.auction_queue.pop(0)
            self._start_auction(next_tile, self.auction_return)
        elif self.auction_return == "advance_turn":
            self._advance_turn()
        elif self.auction_return == "post_turn":
            self.actor = self.active
            self.phase = "post_turn"
        else:
            self._after_landing()

    def _management_actions(self, player: int) -> list[Action]:
        actions: list[Action] = []
        for tile in DEED_TILES:
            if int(self.owner[tile]) != player:
                continue
            if tile not in self.managed_tiles and self.houses[tile] == 0 and not self.mortgaged[tile] and not self._group_has_buildings(tile):
                actions.append(Action(ActionKind.MORTGAGE, tile=tile))
            if tile not in self.managed_tiles and self.mortgaged[tile]:
                cost = int(np.ceil(MORTGAGES[tile] * 1.1))
                if self.cash[player] >= cost:
                    actions.append(Action(ActionKind.UNMORTGAGE, tile=tile, cash=cost))
            if tile in GROUP_OF and tile not in self.sold_tiles and self._can_build(player, tile):
                actions.append(Action(ActionKind.BUILD, tile=tile, cash=int(BUILD_COST[tile])))
            if tile in GROUP_OF and tile not in self.built_tiles and self._can_sell_building(player, tile):
                actions.append(Action(ActionKind.SELL_BUILDING, tile=tile, cash=int(BUILD_COST[tile] // 2)))
        return actions

    def _can_build(self, player: int, tile: int) -> bool:
        group_tiles = GROUPS[GROUP_OF[tile]]
        level = int(self.houses[tile])
        if not self._owns_group(player, tile) or any(self.mortgaged[item] for item in group_tiles):
            return False
        if level >= 5 or level != min(int(self.houses[item]) for item in group_tiles):
            return False
        if self.cash[player] < BUILD_COST[tile]:
            return False
        return self.bank_houses > 0 if level < 4 else self.bank_hotels > 0

    def _can_sell_building(self, player: int, tile: int) -> bool:
        group_tiles = GROUPS[GROUP_OF[tile]]
        level = int(self.houses[tile])
        if int(self.owner[tile]) != player or level <= 0:
            return False
        if level != max(int(self.houses[item]) for item in group_tiles):
            return False
        return level < 5 or self.bank_houses >= 4

    def _build(self, tile: int) -> None:
        player = self.actor
        level = int(self.houses[tile])
        self.cash[player] -= int(BUILD_COST[tile])
        if level == 4:
            self.bank_hotels -= 1
            self.bank_houses += 4
        else:
            self.bank_houses -= 1
        self.houses[tile] += 1

    def _sell_building(self, tile: int) -> None:
        player = self.actor
        level = int(self.houses[tile])
        self.cash[player] += int(BUILD_COST[tile] // 2)
        if level == 5:
            self.bank_hotels += 1
            self.bank_houses -= 4
        else:
            self.bank_houses += 1
        self.houses[tile] -= 1

    def _trade_actions(self, proposer: int) -> list[Action]:
        cash_grid = (0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1200)
        own = [tile for tile in DEED_TILES if int(self.owner[tile]) == proposer and self._tradeable(tile)]
        actions: list[Action] = []
        for target in range(self.player_count):
            if target == proposer or not self.alive[target]:
                continue
            theirs = [tile for tile in DEED_TILES if int(self.owner[tile]) == target and self._tradeable(tile)]
            for tile in theirs:
                for cash in cash_grid:
                    if 0 < cash <= int(self.cash[proposer]):
                        actions.append(Action(ActionKind.TRADE, target=target, cash=cash, take_tile=tile))
            for tile in own:
                for cash in cash_grid:
                    if 0 < cash <= int(self.cash[target]):
                        actions.append(Action(ActionKind.TRADE, target=target, cash=-cash, give_tile=tile))
            for give_tile in own:
                for take_tile in theirs:
                    actions.append(Action(ActionKind.TRADE, target=target, give_tile=give_tile, take_tile=take_tile))
            if self.jail_cards[target] > 0:
                for cash in cash_grid[1:8]:
                    if cash <= int(self.cash[proposer]):
                        actions.append(Action(ActionKind.TRADE, target=target, cash=cash, take_card=1))
            if self.jail_cards[proposer] > 0:
                for cash in cash_grid[1:8]:
                    if cash <= int(self.cash[target]):
                        actions.append(Action(ActionKind.TRADE, target=target, cash=-cash, give_card=1))

        filtered = [action for action in actions if action.key() not in self.rejected_offers]
        if len(filtered) <= 160:
            return filtered
        indices = np.linspace(0, len(filtered) - 1, 160, dtype=np.int32)
        return [filtered[int(index)] for index in indices]

    def _tradeable(self, tile: int) -> bool:
        return tile not in self.traded_tiles and not self._group_has_buildings(tile)

    def _resolve_trade(self, accepted: bool) -> None:
        trade = self.pending_trade
        if trade is None:
            raise RuntimeError("trade response without a pending trade")
        proposer = self.active
        responder = trade.target
        if accepted and self._trade_still_valid(trade, proposer, responder):
            proposer_interest = self._incoming_mortgage_interest(trade.take_tile)
            responder_interest = self._incoming_mortgage_interest(trade.give_tile)
            if trade.cash > 0:
                self.cash[proposer] -= trade.cash
                self.cash[responder] += trade.cash
            elif trade.cash < 0:
                self.cash[proposer] += -trade.cash
                self.cash[responder] -= -trade.cash
            self.cash[proposer] -= proposer_interest
            self.cash[responder] -= responder_interest
            if trade.give_tile >= 0:
                self.owner[trade.give_tile] = responder
                self.traded_tiles.add(trade.give_tile)
            if trade.take_tile >= 0:
                self.owner[trade.take_tile] = proposer
                self.traded_tiles.add(trade.take_tile)
            if trade.give_card:
                self._transfer_jail_cards(proposer, responder, trade.give_card)
            if trade.take_card:
                self._transfer_jail_cards(responder, proposer, trade.take_card)
        else:
            self.rejected_offers.add(trade.key())
        self.trades_left -= 1
        self.pending_trade = None
        self.actor = self.active
        if self.trade_return_phase == "debt" and self.cash[self.active] >= 0:
            self.debt_creditor = -1
            if self.payment_queue:
                self._continue_payments()
            elif self.resume_landing_after_debt:
                self.resume_landing_after_debt = False
                self._resolve_landing(self.active, no_extra_turn=True)
            else:
                self.phase = "post_turn"
        else:
            self.phase = self.trade_return_phase

    def _trade_still_valid(self, trade: Action, proposer: int, responder: int) -> bool:
        proposer_post_cash = int(self.cash[proposer]) - max(0, trade.cash) + max(0, -trade.cash)
        responder_post_cash = int(self.cash[responder]) - max(0, -trade.cash) + max(0, trade.cash)
        if proposer_post_cash < self._incoming_mortgage_interest(trade.take_tile):
            return False
        if responder_post_cash < self._incoming_mortgage_interest(trade.give_tile):
            return False
        if trade.give_tile >= 0 and int(self.owner[trade.give_tile]) != proposer:
            return False
        if trade.take_tile >= 0 and int(self.owner[trade.take_tile]) != responder:
            return False
        if trade.give_card > self.jail_cards[proposer] or trade.take_card > self.jail_cards[responder]:
            return False
        return True

    def _incoming_mortgage_interest(self, tile: int) -> int:
        if tile < 0 or not self.mortgaged[tile]:
            return 0
        return int(math.ceil(MORTGAGES[tile] * 0.1))

    def _transfer_jail_cards(self, source: int, target: int, count: int) -> None:
        for _ in range(count):
            self.jail_card_decks[target].append(self.jail_card_decks[source].pop(0))
        self.jail_cards[source] -= count
        self.jail_cards[target] += count

    def _after_management(self) -> None:
        if self.phase == "debt" and self.cash[self.actor] >= 0:
            self.debt_creditor = -1
            if self.payment_queue:
                self._continue_payments()
            elif self.resume_landing_after_debt:
                self.resume_landing_after_debt = False
                self._resolve_landing(self.active, no_extra_turn=True)
            else:
                self.phase = "post_turn"
                self.actor = self.active

    def _enter_debt(self, debtor: int, creditor: int) -> None:
        self.actor = debtor
        self.debt_creditor = creditor
        self.phase = "debt"

    def _can_raise_cash(self, player: int) -> bool:
        return any(action.kind in (ActionKind.MORTGAGE, ActionKind.SELL_BUILDING, ActionKind.TRADE) for action in self._management_actions(player) + (self._trade_actions(player) if player == self.active else []))

    def _bankrupt(self, player: int, creditor: int) -> None:
        if not self.alive[player]:
            return
        self.alive[player] = False
        self.eliminated_order.append(player)
        bank_auctions: list[int] = []
        if creditor >= 0 and self.alive[creditor]:
            # The creditor was credited when the debt was created. A negative
            # balance is the uncollectible part and must be clawed back here.
            self.cash[creditor] += int(self.cash[player])
            for tile in DEED_TILES:
                if int(self.owner[tile]) == player:
                    level = int(self.houses[tile])
                    if level == 5:
                        self.bank_hotels += 1
                    else:
                        self.bank_houses += level
                    self.cash[creditor] += level * int(BUILD_COST[tile] // 2)
                    self.owner[tile] = creditor
                    self.houses[tile] = 0
                    if self.mortgaged[tile]:
                        self.cash[creditor] -= int(math.ceil(MORTGAGES[tile] * 0.1))
            self._transfer_jail_cards(player, creditor, int(self.jail_cards[player]))
        else:
            for tile in DEED_TILES:
                if int(self.owner[tile]) == player:
                    level = int(self.houses[tile])
                    if level == 5:
                        self.bank_hotels += 1
                    else:
                        self.bank_houses += level
                    self.owner[tile] = -1
                    self.houses[tile] = 0
                    self.mortgaged[tile] = False
                    bank_auctions.append(tile)
            while self.jail_card_decks[player]:
                self._return_jail_card(player)
        self.cash[player] = 0
        if int(self.alive.sum()) <= 1:
            self._finish_game()
            return
        if bank_auctions:
            self.auction_queue = bank_auctions[1:]
            return_phase = "advance_turn" if player == self.active else "post_turn"
            self._start_auction(bank_auctions[0], return_phase)
            return
        if player == self.active:
            self.payment_queue.clear()
            self._advance_turn()
        elif self.payment_queue:
            self._continue_payments()
        else:
            self.actor = self.active
            self.phase = "post_turn"

    def _advance_turn(self) -> None:
        if self.done:
            return
        previous = self.active
        for _ in range(self.player_count):
            self.active = (self.active + 1) % self.player_count
            if self.alive[self.active]:
                break
        if self.active <= previous:
            self.round += 1
        self.turn_count += 1
        if self.round > self.max_rounds:
            self._finish_game(truncated=True)
            return
        self.actor = self.active
        self.phase = "jail" if self.in_jail[self.active] else "turn_start"
        self.doubles_count = 0
        self.trades_left = 6
        self.traded_tiles.clear()
        self.managed_tiles.clear()
        self.built_tiles.clear()
        self.sold_tiles.clear()
        self.rejected_offers.clear()
        self.debt_creditor = -1
        self.resume_landing_after_debt = False

    def _send_to_jail(self, player: int) -> None:
        self.position[player] = 10
        self.in_jail[player] = True
        self.jail_turns[player] = 0
        self.doubles_count = 0

    def _leave_jail(self, player: int) -> None:
        self.in_jail[player] = False
        self.jail_turns[player] = 0

    def _owns_group(self, player: int, tile: int) -> bool:
        if tile not in GROUP_OF:
            return False
        return all(int(self.owner[item]) == player for item in GROUPS[GROUP_OF[tile]])

    def _group_has_buildings(self, tile: int) -> bool:
        if tile not in GROUP_OF:
            return False
        return any(int(self.houses[item]) > 0 for item in GROUPS[GROUP_OF[tile]])

    def _finish_game(self, truncated: bool = False) -> None:
        self.done = True
        self.phase = "done"
        alive = [player for player in range(self.player_count) if self.alive[player]]
        if truncated and len(alive) > 1:
            alive.sort(key=self._liquidation_score)
            elimination = self.eliminated_order + alive
        else:
            elimination = self.eliminated_order + alive
        rank = {player: self.player_count - index for index, player in enumerate(elimination)}
        if self.player_count == 1:
            self.terminal_rewards[:] = 0
        else:
            for player in range(self.player_count):
                normalized = (rank[player] - 1) / (self.player_count - 1)
                self.terminal_rewards[player] = 1.0 - 2.0 * normalized

    def _liquidation_score(self, player: int) -> int:
        total = int(self.cash[player])
        for tile in DEED_TILES:
            if int(self.owner[tile]) != player:
                continue
            if not self.mortgaged[tile]:
                total += int(MORTGAGES[tile])
            total += int(self.houses[tile]) * int(BUILD_COST[tile] // 2)
        return total

    def observe(self) -> dict[str, np.ndarray]:
        return encode_observation(self)

    @staticmethod
    def action_features(actions: Iterable[Action], actor: int, player_count: int) -> dict[str, np.ndarray]:
        return encode_action_features(actions, actor, player_count)

    @staticmethod
    def _deduplicate(actions: Iterable[Action]) -> list[Action]:
        result: list[Action] = []
        seen: set[tuple[int, ...]] = set()
        for action in actions:
            if action.key() in seen:
                continue
            seen.add(action.key())
            result.append(action)
        return result
