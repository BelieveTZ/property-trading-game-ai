from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .board_rules import indexed_board_rules


GLOBAL_FEATURES = 37
PLAYER_FEATURES = 11
PROPERTY_FEATURES = 17
ACTION_NUMERIC_FEATURES = 7
MAX_PLAYERS = 5

_BOARD_RULES = indexed_board_rules()
DEED_TILES = _BOARD_RULES["deed_tiles"]
DEED_INDEX = {tile: index for index, tile in enumerate(DEED_TILES)}
DEED_PADDING = len(DEED_TILES)
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
PHASES = (
    "turn_start", "jail", "buy", "auction", "post_turn", "debt", "trade_response", "done"
)
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}


def collate_observations(
    observations: Sequence[dict[str, object]],
    device: torch.device | str,
) -> dict[str, Tensor]:
    return {
        key: torch.as_tensor(np.stack([observation[key] for observation in observations]), device=device)
        for key in ("global", "players", "properties")
    }


def encode_observation(state: Any) -> dict[str, np.ndarray]:
    actor = state.actor
    player_count = state.player_count
    order = [(actor + offset) % player_count for offset in range(player_count)]
    global_features = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
    global_features[0] = player_count / MAX_PLAYERS
    global_features[1 + PHASE_INDEX[state.phase]] = 1.0
    global_features[9] = min(state.round / state.max_rounds, 1.0)
    global_features[10] = state.last_roll / 12.0
    global_features[11] = float(state.last_doubles)
    global_features[12] = state.doubles_count / 3.0
    global_features[13] = state.bank_houses / 32.0
    global_features[14] = state.bank_hotels / 12.0
    global_features[15] = state.trades_left / 6.0
    global_features[16] = max(-1.0, min(2.0, float(state.cash[actor]) / 1500.0))
    global_features[17] = state.pending_tile / 39.0 if state.pending_tile >= 0 else -1.0
    if state.auction:
        global_features[18] = int(state.auction["bid"]) / 1500.0
        global_features[19] = int(state.auction["tile"]) / 39.0
    global_features[20] = max(0.0, -float(state.cash[actor]) / 1500.0)
    global_features[21] = float(actor == state.active)
    global_features[22] = int(state.position[actor]) / 39.0
    global_features[23] = float(state.in_jail[actor])
    global_features[24] = order.index(state.active) / max(1, player_count - 1)
    if state.auction:
        leader = int(state.auction["leader"])
        global_features[25] = order.index(leader) / max(1, player_count - 1) if leader >= 0 else -1.0
        global_features[26] = len(state.auction["active"]) / player_count
    else:
        global_features[25] = -1.0
    global_features[27] = order.index(state.debt_creditor) / max(1, player_count - 1) if state.debt_creditor >= 0 else -1.0
    if state.pending_trade is not None:
        trade = state.pending_trade
        global_features[28] = trade.cash / 1500.0
        global_features[29] = trade.give_tile / 39.0 if trade.give_tile >= 0 else -1.0
        global_features[30] = trade.take_tile / 39.0 if trade.take_tile >= 0 else -1.0
        global_features[31] = float(trade.give_card)
        global_features[32] = float(trade.take_card)
        global_features[35] = 1.0
    else:
        global_features[29:31] = -1.0
    global_features[33] = min(len(state.rejected_offers) / 6.0, 1.0)
    global_features[34] = min(state.turn_count / max(1, state.max_rounds * player_count), 1.0)
    global_features[36] = min(sum(amount for _, _, amount in state.payment_queue) / 1500.0, 1.0)

    players = np.zeros((MAX_PLAYERS, PLAYER_FEATURES), dtype=np.float32)
    for slot, player in enumerate(order):
        players[slot] = (
            1.0,
            float(state.alive[player]),
            float(player == state.active),
            float(player == actor),
            np.clip(state.cash[player] / 3000.0, -1.0, 2.0),
            state.position[player] / 39.0,
            float(state.in_jail[player]),
            state.jail_turns[player] / 3.0,
            state.jail_cards[player] / 2.0,
            sum(int(state.owner[tile]) == player for tile in DEED_TILES) / len(DEED_TILES),
            slot / max(1, player_count - 1),
        )

    properties = np.zeros((len(DEED_TILES), PROPERTY_FEATURES), dtype=np.float32)
    for index, tile in enumerate(DEED_TILES):
        owner = int(state.owner[tile])
        relative_owner = order.index(owner) if owner >= 0 else -1
        rents = RENT_TABLE.get(tile, (0, 0, 0, 0, 0, 0))
        properties[index] = (
            tile / 39.0,
            1.0 if tile in GROUP_OF else 0.0,
            1.0 if tile in RAILROADS else 0.0,
            1.0 if tile in UTILITIES else 0.0,
            (GROUP_OF.get(tile, -1) + 1) / 8.0,
            (relative_owner + 1) / 6.0,
            state.houses[tile] / 5.0,
            float(state.mortgaged[tile]),
            PRICES[tile] / 400.0,
            MORTGAGES[tile] / 200.0,
            BUILD_COST[tile] / 200.0,
            rents[0] / 50.0,
            rents[1] / 200.0,
            rents[2] / 600.0,
            rents[3] / 1400.0,
            rents[4] / 1700.0,
            rents[5] / 2000.0,
        )
    return {"global": global_features, "players": players, "properties": properties}


def encode_action_features(
    actions: Iterable[Any],
    actor: int,
    player_count: int,
) -> dict[str, np.ndarray]:
    action_list = list(actions)
    kinds = np.array([int(action.kind) for action in action_list], dtype=np.int64)
    tiles = np.array([DEED_INDEX.get(action.tile, DEED_PADDING) for action in action_list], dtype=np.int64)
    give_tiles = np.array([DEED_INDEX.get(action.give_tile, DEED_PADDING) for action in action_list], dtype=np.int64)
    take_tiles = np.array([DEED_INDEX.get(action.take_tile, DEED_PADDING) for action in action_list], dtype=np.int64)
    targets = np.array(
        [
            (action.target - actor) % player_count
            if action.target >= 0
            else MAX_PLAYERS
            for action in action_list
        ],
        dtype=np.int64,
    )
    numeric = np.zeros((len(action_list), ACTION_NUMERIC_FEATURES), dtype=np.float32)
    for index, action in enumerate(action_list):
        numeric[index] = (
            np.clip(action.cash / 1500.0, -1.0, 1.0),
            action.give_tile / 39.0 if action.give_tile >= 0 else -1.0,
            action.take_tile / 39.0 if action.take_tile >= 0 else -1.0,
            float(action.give_card),
            float(action.take_card),
            float(action.tile >= 0),
            float(action.target >= 0),
        )
    return {
        "kind": kinds,
        "tile": tiles,
        "give_tile": give_tiles,
        "take_tile": take_tiles,
        "target": targets,
        "numeric": numeric,
    }


def collate_action_feature_rows(
    feature_rows: Sequence[dict[str, np.ndarray]],
    device: torch.device | str,
) -> tuple[dict[str, Tensor], Tensor]:
    maximum = max(len(features["kind"]) for features in feature_rows)
    batch = len(feature_rows)
    arrays = {
        "kind": np.zeros((batch, maximum), dtype=np.int64),
        "tile": np.full((batch, maximum), DEED_PADDING, dtype=np.int64),
        "give_tile": np.full((batch, maximum), DEED_PADDING, dtype=np.int64),
        "take_tile": np.full((batch, maximum), DEED_PADDING, dtype=np.int64),
        "target": np.full((batch, maximum), MAX_PLAYERS, dtype=np.int64),
        "numeric": np.zeros((batch, maximum, ACTION_NUMERIC_FEATURES), dtype=np.float32),
    }
    mask = np.zeros((batch, maximum), dtype=np.bool_)
    for row, features in enumerate(feature_rows):
        size = len(features["kind"])
        for key in arrays:
            arrays[key][row, :size] = features[key]
        mask[row, :size] = True
    return (
        {key: torch.as_tensor(value, device=device) for key, value in arrays.items()},
        torch.as_tensor(mask, device=device),
    )


def collate_actions(
    action_lists: Sequence[Sequence[Any]],
    actors: Sequence[int],
    player_counts: Sequence[int],
    device: torch.device | str,
) -> tuple[dict[str, Tensor], Tensor]:
    rows = [
        encode_action_features(actions, actor, count)
        for actions, actor, count in zip(action_lists, actors, player_counts, strict=True)
    ]
    return collate_action_feature_rows(rows, device)
