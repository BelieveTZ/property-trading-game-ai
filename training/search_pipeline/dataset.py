from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from training.zero_knowledge.env import PropertyTradingEnv

from .protocol import TEACHER_VERSION, SearchTeacher, public_view


def _sample(
    view: dict[str, object],
    policy: list[float],
    action_index: int,
    *,
    action: object,
    decision_index: int,
    game_seed: int,
    teacher_seed: int,
    simulations: int,
    depth: int,
    max_rounds: int,
) -> dict[str, object]:
    return {
        "version": 1,
        "game_seed": game_seed,
        "teacher_seed": teacher_seed,
        "teacher_version": TEACHER_VERSION,
        "teacher_config": {
            "simulations": simulations,
            "depth": depth,
            "max_rounds": max_rounds,
        },
        "decision_index": decision_index,
        "actor": int(view["actor"]),
        "seat": int(view["seat"]),
        "player_count": int(view["player_count"]),
        "observation": {
            name: value.tolist()
            for name, value in view["observation"].items()
        },
        "legal_actions": [asdict(action) for action in view["legal_actions"]],
        "policy": policy,
        "selected_action": action_index,
        "action": asdict(action),
        "outcome": 0.0,
    }


def generate_teacher_episode(
    *,
    player_count: int,
    game_seed: int,
    teacher_seed: int,
    simulations: int,
    depth: int,
    max_rounds: int = 360,
) -> list[dict[str, object]]:
    env = PropertyTradingEnv(player_count, seed=game_seed, max_rounds=max_rounds)
    teacher = SearchTeacher(seed=teacher_seed, simulations=simulations, depth=depth)
    samples: list[dict[str, object]] = []
    for _ in range(max_rounds * player_count * 300):
        if env.done:
            break
        view = public_view(env, env.actor)
        action, policy = teacher.select_action(env)
        action_index = view["legal_actions"].index(action)
        samples.append(_sample(
            view,
            policy.tolist(),
            action_index,
            action=action,
            decision_index=len(samples),
            game_seed=game_seed,
            teacher_seed=teacher_seed,
            simulations=simulations,
            depth=depth,
            max_rounds=max_rounds,
        ))
        env.step(action)
    if not env.done:
        raise RuntimeError("teacher episode exceeded its deterministic decision budget")
    for sample in samples:
        sample["outcome"] = float(env.terminal_rewards[int(sample["seat"])])
    return samples


def write_teacher_dataset(path: Path, samples: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_teacher_dataset(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
