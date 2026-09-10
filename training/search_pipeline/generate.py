from __future__ import annotations

import argparse
import time
from pathlib import Path

from .control import SearchRunControl
from .config import load_search_config
from .dataset import generate_teacher_episode, read_teacher_dataset, write_teacher_dataset


def teacher_generation_defaults() -> dict[str, int]:
    teacher = load_search_config()["teacher"]
    return {
        "simulations": int(teacher["simulations"]),
        "depth": int(teacher["depth"]),
    }


def parse_args() -> argparse.Namespace:
    defaults = teacher_generation_defaults()
    parser = argparse.ArgumentParser(description="Generate reproducible search-teacher games")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("training/runs/search-main"))
    parser.add_argument("--players", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--game-seed", type=int, default=20260910)
    parser.add_argument("--teacher-seed", type=int, default=20260911)
    parser.add_argument("--simulations", type=int, default=defaults["simulations"])
    parser.add_argument("--depth", type=int, default=defaults["depth"])
    parser.add_argument("--max-rounds", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control = SearchRunControl(args.run_dir)
    state = control.begin()
    samples = read_teacher_dataset(args.output) if args.output.exists() else []
    for game_index in range(state.completed_games, args.games):
        if control.pause_requested():
            control.save(state)
            print(f"paused after {state.completed_games} games")
            return
        started = time.perf_counter()
        samples.extend(generate_teacher_episode(
            player_count=args.players,
            game_seed=args.game_seed + game_index,
            teacher_seed=args.teacher_seed + game_index,
            simulations=args.simulations,
            depth=args.depth,
            max_rounds=args.max_rounds,
        ))
        state.completed_games = game_index + 1
        state.gpu_seconds += time.perf_counter() - started
        write_teacher_dataset(args.output, samples)
        control.save(state)
    print(f"wrote {len(samples)} decisions from {state.completed_games} games to {args.output}")


if __name__ == "__main__":
    main()
