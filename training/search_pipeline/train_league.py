from __future__ import annotations

import argparse
import json
from pathlib import Path

from .league import run_league_training
from .protocol import LeagueRoster


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a shared policy against frozen league opponents")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opponent", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-rounds", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    roster = LeagueRoster(args.seed)
    for value in args.opponent:
        name, separator, checkpoint = value.partition("=")
        if not separator or not name or not checkpoint:
            raise ValueError("opponents must use NAME=CHECKPOINT")
        roster.add(name, checkpoint, frozen=True)
    metrics = run_league_training(
        checkpoint=args.checkpoint,
        roster=roster,
        output=args.output,
        games=args.games,
        player_counts=tuple(int(value) for value in args.players.split(",")),
        seed=args.seed,
        max_rounds=args.max_rounds,
        device=args.device,
    )
    roster.save(args.output.with_suffix(".league.json"))
    print(json.dumps({"output": str(args.output), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
