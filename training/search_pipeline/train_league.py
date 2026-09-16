from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .control import record_training_command
from .league import run_league_training
from .protocol import LeagueRoster


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a shared policy against frozen league opponents")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--opponent", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-rounds", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-hours", type=float)
    args = parser.parse_args()
    run_directory = args.run_dir or args.output.parent
    record_training_command(
        run_directory,
        module="training.search_pipeline.train_league",
        arguments=sys.argv[1:],
        pid_file="league-pid.txt",
        pause_file="league-pause.request",
        state_file="league-state.json",
        stdout_log="league.stdout.log",
        stderr_log="league.stderr.log",
    )
    pause_path = run_directory / "league-pause.request"
    pause_path.unlink(missing_ok=True)
    (run_directory / "league-pid.txt").write_text(str(os.getpid()), encoding="ascii")
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
        run_directory=run_directory,
        should_pause=lambda _game, _decision: pause_path.exists(),
        gpu_hours=args.gpu_hours,
    )
    if metrics.get("status") == "paused":
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    roster.save(args.output.with_suffix(".league.json"))
    print(json.dumps({"output": str(args.output), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
