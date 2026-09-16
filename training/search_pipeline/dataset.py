from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from training.zero_knowledge.env import PropertyTradingEnv

from .protocol import TEACHER_VERSION, SearchTeacher, public_view


class TeacherEpisodePaused(RuntimeError):
    """Raised after an in-progress teacher game has been checkpointed safely."""


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
        candidate_indices = np.flatnonzero(policy > 0)
        candidate_actions = [
            view["legal_actions"][int(index)]
            for index in candidate_indices
        ]
        candidate_policy = policy[candidate_indices]
        candidate_policy /= candidate_policy.sum()
        candidate_view = dict(view)
        candidate_view["legal_actions"] = candidate_actions
        action_index = candidate_actions.index(action)
        samples.append(_sample(
            candidate_view,
            candidate_policy.tolist(),
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


def generate_teacher_episode_file(
    *,
    output: Path,
    checkpoint: Path,
    player_count: int,
    game_seed: int,
    teacher_seed: int,
    simulations: int,
    depth: int,
    max_rounds: int = 360,
    checkpoint_every: int = 32,
    should_pause: Callable[[int], bool] | None = None,
) -> int:
    """Generate one game atomically with resumable intra-game progress."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if output.exists():
        return len(read_teacher_dataset(output))

    signature = {
        "version": 1,
        "player_count": player_count,
        "game_seed": game_seed,
        "teacher_seed": teacher_seed,
        "simulations": simulations,
        "depth": depth,
        "max_rounds": max_rounds,
    }
    partial = output.with_suffix(".partial.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if saved.get("signature") != signature:
            raise ValueError("teacher episode checkpoint does not match requested configuration")
        env = saved["env"]
        teacher = saved["teacher"]
        decision_count = int(saved["decision_count"])
        samples = read_teacher_dataset(partial) if partial.exists() else []
        if len(samples) < decision_count:
            raise ValueError("teacher episode checkpoint is ahead of its partial dataset")
        if len(samples) > decision_count:
            samples = samples[:decision_count]
            write_teacher_dataset(partial, samples)
    else:
        partial.unlink(missing_ok=True)
        env = PropertyTradingEnv(player_count, seed=game_seed, max_rounds=max_rounds)
        teacher = SearchTeacher(seed=teacher_seed, simulations=simulations, depth=depth)
        decision_count = 0
        samples: list[dict[str, object]] = []

    def save_checkpoint() -> None:
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump({
                "signature": signature,
                "env": env,
                "teacher": teacher,
                "decision_count": decision_count,
            }, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(checkpoint)

    with partial.open("a", encoding="utf-8", newline="\n") as stream:
        for _ in range(decision_count, max_rounds * player_count * 300):
            if env.done:
                break
            view = public_view(env, env.actor)
            action, policy = teacher.select_action(env)
            candidate_indices = np.flatnonzero(policy > 0)
            candidate_actions = [
                view["legal_actions"][int(index)]
                for index in candidate_indices
            ]
            candidate_policy = policy[candidate_indices]
            candidate_policy /= candidate_policy.sum()
            candidate_view = dict(view)
            candidate_view["legal_actions"] = candidate_actions
            action_index = candidate_actions.index(action)
            sample = _sample(
                candidate_view,
                candidate_policy.tolist(),
                action_index,
                action=action,
                decision_index=decision_count,
                game_seed=game_seed,
                teacher_seed=teacher_seed,
                simulations=simulations,
                depth=depth,
                max_rounds=max_rounds,
            )
            stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            samples.append(sample)
            env.step(action)
            decision_count += 1

            pause = should_pause is not None and should_pause(decision_count)
            if decision_count % checkpoint_every == 0 or pause:
                stream.flush()
                os.fsync(stream.fileno())
                save_checkpoint()
            if pause:
                raise TeacherEpisodePaused(
                    f"paused teacher episode after {decision_count} decisions",
                )

    if not env.done:
        raise RuntimeError("teacher episode exceeded its deterministic decision budget")
    for sample in samples:
        sample["outcome"] = float(env.terminal_rewards[int(sample["seat"])])
    write_teacher_dataset(output, samples)
    partial.unlink(missing_ok=True)
    checkpoint.unlink(missing_ok=True)
    return decision_count


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
