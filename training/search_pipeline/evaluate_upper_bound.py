from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import (
    _write_json_atomic,
    build_balanced_evaluation_schedule,
    checkpoint_artifact,
    evaluate_policy_league,
    parse_named_checkpoints,
)
from .policies import load_frozen_policy
from .upper_bound import ConfiguredSearchPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the configured high-budget search policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opponent", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--seeds", default="20260910,20260911,20260912,20260913")
    parser.add_argument("--teacher-seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rounds", type=int, default=360)
    args = parser.parse_args()

    policy = ConfiguredSearchPolicy(seed=args.teacher_seed)
    opponents = {}
    opponent_artifacts = []
    for name, checkpoint_path in parse_named_checkpoints(
        args.opponent,
        candidate_name=policy.version,
    ):
        opponents[name] = load_frozen_policy(checkpoint_path, device=args.device)
        opponent_artifacts.append(checkpoint_artifact(checkpoint_path, name=name))
    schedule = build_balanced_evaluation_schedule(
        player_counts=tuple(int(value) for value in args.players.split(",")),
        seeds=tuple(int(value) for value in args.seeds.split(",")),
        candidate=policy.version,
        opponents=tuple(opponents),
    )
    report = evaluate_policy_league(
        schedule,
        {policy.version: policy, **opponents},
        max_rounds=args.max_rounds,
    )
    report["searchPolicy"] = policy.provenance()
    report["opponentArtifacts"] = opponent_artifacts
    _write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
