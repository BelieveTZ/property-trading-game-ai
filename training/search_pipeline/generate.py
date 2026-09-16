from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .control import SearchRunControl, SearchRunState, record_training_command
from .config import load_search_config
from .dataset import TeacherEpisodePaused, generate_teacher_episode_file


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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 2)),
    )
    parser.add_argument("--checkpoint-every", type=int, default=32)
    return parser.parse_args()


def _generate_game(job: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    pause_path = Path(str(job["pause_path"]))
    if pause_path.exists():
        return {"game_index": int(job["game_index"]), "complete": False, "cpu_seconds": 0.0}
    try:
        decisions = generate_teacher_episode_file(
            output=Path(str(job["output"])),
            checkpoint=Path(str(job["checkpoint"])),
            player_count=int(job["player_count"]),
            game_seed=int(job["game_seed"]),
            teacher_seed=int(job["teacher_seed"]),
            simulations=int(job["simulations"]),
            depth=int(job["depth"]),
            max_rounds=int(job["max_rounds"]),
            checkpoint_every=int(job["checkpoint_every"]),
            should_pause=lambda _decisions: pause_path.exists(),
        )
    except TeacherEpisodePaused:
        return {
            "game_index": int(job["game_index"]),
            "complete": False,
            "cpu_seconds": time.perf_counter() - started,
        }
    return {
        "game_index": int(job["game_index"]),
        "complete": True,
        "decisions": decisions,
        "cpu_seconds": time.perf_counter() - started,
    }


def _merge_shards(output: Path, shards: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for shard in shards:
            with shard.open("rb") as source:
                shutil.copyfileobj(source, destination)
    temporary.replace(output)


def generate_teacher_games(
    *,
    output: Path,
    run_directory: Path,
    player_count: int,
    games: int,
    game_seed: int,
    teacher_seed: int,
    simulations: int,
    depth: int,
    max_rounds: int,
    workers: int,
    checkpoint_every: int,
) -> SearchRunState:
    if games < 1 or workers < 1:
        raise ValueError("games and workers must be positive")
    control = SearchRunControl(run_directory)
    state = control.begin()
    shards_directory = run_directory / "games"
    shards_directory.mkdir(parents=True, exist_ok=True)
    shards = [shards_directory / f"game-{index:05d}.jsonl" for index in range(games)]
    pending = [index for index, shard in enumerate(shards) if not shard.exists()]
    state.completed_games = games - len(pending)
    control.save(state)

    jobs = [{
        "game_index": index,
        "output": str(shards[index]),
        "checkpoint": str(shards[index].with_suffix(".checkpoint")),
        "pause_path": str(control.pause_path),
        "player_count": player_count,
        "game_seed": game_seed + index,
        "teacher_seed": teacher_seed + index,
        "simulations": simulations,
        "depth": depth,
        "max_rounds": max_rounds,
        "checkpoint_every": checkpoint_every,
    } for index in pending]

    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = [executor.submit(_generate_game, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                state.teacher_cpu_seconds += float(result["cpu_seconds"])
                state.completed_games = sum(shard.exists() for shard in shards)
                control.save(state)

    completed_shards = [shard for shard in shards if shard.exists()]
    if completed_shards:
        _merge_shards(output, completed_shards)
    state.status = "paused" if control.pause_requested() else "completed"
    control.save(state)
    return state


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    record_training_command(
        args.run_dir,
        module="training.search_pipeline.generate",
        arguments=sys.argv[1:],
        pid_file="launcher-pid.txt",
        pause_file="pause.request",
        state_file="search-run.json",
        stdout_log="trainer.stdout.log",
        stderr_log="trainer.stderr.log",
    )
    (args.run_dir / "launcher-pid.txt").write_text(
        str(os.getpid()),
        encoding="ascii",
    )
    state = generate_teacher_games(
        output=args.output,
        run_directory=args.run_dir,
        player_count=args.players,
        games=args.games,
        game_seed=args.game_seed,
        teacher_seed=args.teacher_seed,
        simulations=args.simulations,
        depth=args.depth,
        max_rounds=args.max_rounds,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )
    status = "paused" if SearchRunControl(args.run_dir).pause_requested() else "completed"
    print(f"{status} with {state.completed_games}/{args.games} teacher games in {args.run_dir}")


if __name__ == "__main__":
    main()
