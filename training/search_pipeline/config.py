from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_search_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1:
        raise ValueError("unsupported search pipeline config")
    if not value.get("teacher", {}).get("version"):
        raise ValueError("teacher version is required")
    if not value.get("student", {}).get("version"):
        raise ValueError("student version is required")
    return value
