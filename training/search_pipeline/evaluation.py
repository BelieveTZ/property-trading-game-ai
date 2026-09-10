from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from training.zero_knowledge.env import Action, PropertyTradingEnv

from .control import SearchRunState
from .protocol import EvaluationSchedule, public_view
from .policies import load_frozen_policy

Policy = Callable[[dict[str, object]], Action]


@dataclass(frozen=True, slots=True)
class EvaluationGame:
    player_count: int
    seed: int
    candidate: str
    candidate_seat: int
    seat_policies: tuple[str, ...]


def build_balanced_evaluation_schedule(
    *,
    player_counts: Iterable[int],
    seeds: Iterable[int],
    candidate: str,
    opponents: tuple[str, ...],
) -> list[EvaluationGame]:
    if not opponents:
        raise ValueError("at least one opponent is required")
    schedule: list[EvaluationGame] = []
    for player_count in player_counts:
        if player_count not in (3, 4, 5):
            raise ValueError("player_count must be 3, 4, or 5")
        for seed in seeds:
            for candidate_seat in range(player_count):
                for opponent_rotation in range(len(opponents)):
                    seats = []
                    opponent_index = opponent_rotation
                    for seat in range(player_count):
                        if seat == candidate_seat:
                            seats.append(candidate)
                        else:
                            seats.append(opponents[opponent_index % len(opponents)])
                            opponent_index += 1
                    schedule.append(EvaluationGame(
                        player_count=player_count,
                        seed=int(seed),
                        candidate=candidate,
                        candidate_seat=candidate_seat,
                        seat_policies=tuple(seats),
                    ))
    return schedule


def _wilson95(wins: int, games: int) -> list[float]:
    if games == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * games)) / games) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate_policy_league(
    schedule: Iterable[EvaluationGame],
    policies: dict[str, Policy],
    *,
    max_rounds: int = 360,
) -> dict[str, object]:
    games = list(schedule)
    wins = 0
    ranks: list[int] = []
    decision_times: list[float] = []
    illegal_actions = 0
    for assignment in games:
        missing = set(assignment.seat_policies) - policies.keys()
        if missing:
            raise ValueError(f"missing policies: {sorted(missing)}")
        env = PropertyTradingEnv(
            assignment.player_count,
            seed=assignment.seed,
            max_rounds=max_rounds,
        )
        for policy_name in set(assignment.seat_policies):
            reset = getattr(policies[policy_name], "start_game", None)
            if callable(reset):
                reset(assignment)
        decision_budget = max_rounds * assignment.player_count * 300
        for _ in range(decision_budget):
            if env.done:
                break
            seat = env.actor
            view = public_view(env, seat)
            started = time.perf_counter_ns()
            policy = policies[assignment.seat_policies[seat]]
            if hasattr(policy, "select_action"):
                action = policy.select_action(env)
            else:
                action = policy(view)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if seat == assignment.candidate_seat:
                decision_times.append(elapsed_ms)
            legal = view["legal_actions"]
            if action not in legal:
                illegal_actions += 1
                action = legal[0]
            env.step(action)
        if not env.done:
            raise RuntimeError("evaluation game exceeded its deterministic decision budget")
        candidate_reward = float(env.terminal_rewards[assignment.candidate_seat])
        rank = 1 + sum(float(reward) > candidate_reward for reward in env.terminal_rewards)
        ranks.append(rank)
        wins += int(rank == 1)
    ordered_times = sorted(decision_times)
    p95_index = max(0, math.ceil(len(ordered_times) * 0.95) - 1)
    candidate = games[0].candidate if games else None
    return {
        "version": 1,
        "candidate": candidate,
        "games": len(games),
        "assignments": [asdict(game) for game in games],
        "winRate": wins / len(games) if games else 0.0,
        "averageRank": float(np.mean(ranks)) if ranks else 0.0,
        "wilson95": _wilson95(wins, len(games)),
        "decisionTimeMs": {
            "mean": float(np.mean(decision_times)) if decision_times else 0.0,
            "p95": ordered_times[p95_index] if ordered_times else 0.0,
        },
        "illegalActions": illegal_actions,
    }


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def checkpoint_artifact(path: Path, *, name: str) -> dict[str, object]:
    """Describe the exact frozen policy file used by an evaluation."""
    resolved = path.resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    files: list[dict[str, object]] = []
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        candidates = [resolved]
        if (
            payload.get("version") == "league-neuroevolution-v5-marginal-trade-valuation"
            and not resolved.stem.endswith(("-3p", "-5p"))
        ):
            candidates.extend(
                resolved.with_name(f"{resolved.stem}-{player_count}p{resolved.suffix}")
                for player_count in (3, 5)
            )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
            files.append({
                "playerCount": int(candidate_payload.get("playerCount", 4)),
                "path": str(candidate.resolve()),
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            })
        files.sort(key=lambda item: int(item["playerCount"]))
    else:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    version = payload.get("policy_version") or payload.get("version") or resolved.stem
    artifact: dict[str, object] = {
        "name": name,
        "path": str(resolved),
        "sha256": digest,
        "policyVersion": str(version),
    }
    if files:
        artifact["files"] = files
    return artifact


def parse_named_checkpoints(
    values: Iterable[str],
    *,
    candidate_name: str,
) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        name, separator, checkpoint = value.partition("=")
        if not separator or not name or not checkpoint:
            raise ValueError("opponents must use NAME=CHECKPOINT")
        if name == candidate_name:
            raise ValueError(f"opponent name {name!r} conflicts with candidate")
        if name in seen:
            raise ValueError(f"duplicate opponent name: {name}")
        seen.add(name)
        parsed.append((name, Path(checkpoint)))
    return parsed


def freeze_due_milestones(
    *,
    state: SearchRunState,
    checkpoint: Path,
    run_directory: Path,
    report: dict[str, object],
) -> list[dict[str, object]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    schedule = EvaluationSchedule(completed=set(state.completed_milestones))
    due = schedule.due(state.gpu_seconds / 3600)
    if len(due) > 1:
        raise RuntimeError(
            f"missed evaluation milestones: {due[:-1]}; restore the corresponding checkpoints",
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frozen: list[dict[str, object]] = []
    for milestone in due:
        milestone_directory = run_directory / "milestones" / f"{milestone}-gpu-hours"
        frozen_checkpoint = milestone_directory / "candidate.pt"
        enriched = {
            **report,
            "milestoneGpuHours": milestone,
            "checkpointSha256": digest,
        }
        _copy_atomic(checkpoint, frozen_checkpoint)
        _write_json_atomic(milestone_directory / "evaluation.json", enriched)
        frozen.append(enriched)
    state.completed_milestones = sorted(schedule.completed)

    best_path = run_directory / "best-evaluation.json"
    previous = json.loads(best_path.read_text("utf-8")) if best_path.exists() else None
    eligible = report.get("illegalActions") == 0
    better = previous is None or (
        float(report.get("winRate", 0.0)),
        -float(report.get("averageRank", math.inf)),
    ) > (
        float(previous.get("winRate", 0.0)),
        -float(previous.get("averageRank", math.inf)),
    )
    if due and eligible and better:
        best_report = {**frozen[-1], "sourceCheckpoint": str(checkpoint)}
        _copy_atomic(checkpoint, run_directory / "best-candidate.pt")
        _write_json_atomic(best_path, best_report)
    return frozen


def _checkpoint_policy(path: Path, device_name: str) -> Policy:
    return load_frozen_policy(path, device=device_name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run balanced fixed-seed evaluation and freeze due milestones",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("training/runs/search-main"))
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--seeds", default="20260910,20260911,20260912,20260913")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rounds", type=int, default=360)
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        help="Frozen opponent as NAME=CHECKPOINT; may be repeated",
    )
    args = parser.parse_args()

    player_counts = tuple(int(value) for value in args.players.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    candidate_artifact = checkpoint_artifact(args.checkpoint, name="candidate")
    candidate_name = str(candidate_artifact["policyVersion"])
    opponent_specs = parse_named_checkpoints(args.opponent, candidate_name=candidate_name)
    opponent_policies: dict[str, Policy] = {}
    for name, opponent_path in opponent_specs:
        opponent_policies[name] = _checkpoint_policy(opponent_path, args.device)
    if not opponent_policies:
        opponent_policies["legal-baseline-v1"] = lambda view: view["legal_actions"][0]
    schedule = build_balanced_evaluation_schedule(
        player_counts=player_counts,
        seeds=seeds,
        candidate=candidate_name,
        opponents=tuple(opponent_policies),
    )
    candidate = _checkpoint_policy(args.checkpoint, args.device)
    report = evaluate_policy_league(
        schedule,
        {
            candidate_name: candidate,
            **opponent_policies,
        },
        max_rounds=args.max_rounds,
    )
    report["candidateArtifact"] = candidate_artifact
    report["opponentArtifacts"] = [
        checkpoint_artifact(path, name=name)
        for name, path in opponent_specs
    ]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.run_dir / "evaluations" / "latest.json", report)
    state_path = args.run_dir / "search-run.json"
    if state_path.exists():
        value = json.loads(state_path.read_text(encoding="utf-8"))
        state = SearchRunState(**value)
    else:
        state = SearchRunState(stage="evaluation")
    state.gpu_seconds = max(state.gpu_seconds, args.gpu_hours * 3600)
    frozen = freeze_due_milestones(
        state=state,
        checkpoint=args.checkpoint,
        run_directory=args.run_dir,
        report=report,
    )
    _write_json_atomic(state_path, asdict(state))
    print(json.dumps({**report, "frozenMilestones": frozen}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
