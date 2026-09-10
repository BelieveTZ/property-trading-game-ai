from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_CATALOG_PATH = Path(__file__).resolve().parents[2] / "shared" / "board-rules.json"


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def board_rule_for(tile: int) -> dict[str, Any]:
    rule = dict(_catalog()["deeds"][str(tile)])
    rule["rents"] = tuple(rule["rents"])
    return rule


@lru_cache(maxsize=1)
def indexed_board_rules() -> dict[str, Any]:
    catalog = _catalog()
    board_size = int(catalog["boardSize"])
    group_order = tuple(catalog["groupOrder"])
    groups = {name: tuple(tiles) for name, tiles in catalog["groups"].items()}
    deeds = {int(tile): board_rule_for(int(tile)) for tile in catalog["deeds"]}

    prices = np.zeros(board_size, dtype=np.int16)
    base_rents = np.zeros(board_size, dtype=np.int16)
    mortgages = np.zeros(board_size, dtype=np.int16)
    build_costs = np.zeros(board_size, dtype=np.int16)
    group_indexes = np.full(board_size, -1, dtype=np.int8)
    for tile, rule in deeds.items():
        prices[tile] = rule["price"]
        base_rents[tile] = rule["baseRent"]
        mortgages[tile] = rule["mortgage"]
        build_costs[tile] = rule["buildCost"]
        group_indexes[tile] = group_order.index(rule["group"])

    return {
        "board_size": board_size,
        "deed_tiles": tuple(sorted(deeds)),
        "prices": prices,
        "base_rents": base_rents,
        "mortgages": mortgages,
        "build_costs": build_costs,
        "group_indexes": group_indexes,
        "group_order": group_order,
        "groups": groups,
        "rent_table": {
            tile: rule["rents"] for tile, rule in deeds.items() if rule["type"] == "property"
        },
    }
