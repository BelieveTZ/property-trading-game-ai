from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path

import numpy as np

from training.zero_knowledge.env import Action, PropertyTradingEnv
from .config import load_search_config


_PIPELINE_CONFIG = load_search_config()
TEACHER_VERSION = str(_PIPELINE_CONFIG["teacher"]["version"])


def _configured_milestones() -> tuple[int, ...]:
    return tuple(int(value) for value in _PIPELINE_CONFIG["league"]["milestoneGpuHours"])


def public_view(env: PropertyTradingEnv, seat: int) -> dict[str, object]:
    """Return the complete policy boundary without simulator-private random state."""
    if seat < 0 or seat >= env.player_count:
        raise ValueError("seat is outside the game")
    observation = {
        name: np.asarray(value).copy()
        for name, value in env.observe().items()
    }
    return {
        "actor": env.actor,
        "seat": seat,
        "player_count": env.player_count,
        "observation": observation,
        "legal_actions": list(env.legal_actions()),
    }


@dataclass(slots=True)
class SeatRuntime:
    seat: int
    seed: int
    rng: np.random.Generator
    memory: dict[str, object] = field(default_factory=dict)
    reward: float = 0.0

    @classmethod
    def create(cls, game_seed: int, seat: int) -> "SeatRuntime":
        sequence = np.random.SeedSequence([int(game_seed), int(seat), 0x5E47])
        seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        return cls(seat=seat, seed=seed, rng=np.random.default_rng(seed))


@dataclass(slots=True)
class EvaluationSchedule:
    milestones: tuple[int, ...] = field(default_factory=_configured_milestones)
    completed: set[int] = field(default_factory=set)

    def due(self, gpu_hours: float) -> list[int]:
        ready = [
            milestone
            for milestone in self.milestones
            if milestone <= gpu_hours and milestone not in self.completed
        ]
        self.completed.update(ready)
        return ready


@dataclass(frozen=True, slots=True)
class LeagueEntry:
    name: str
    checkpoint: str
    frozen: bool


class LeagueRoster:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.entries: list[LeagueEntry] = []

    def add(self, name: str, checkpoint: str, *, frozen: bool) -> None:
        if any(entry.name == name for entry in self.entries):
            raise ValueError(f"league entry already exists: {name}")
        self.entries.append(LeagueEntry(name=name, checkpoint=checkpoint, frozen=frozen))

    def sample(self, count: int) -> list[str]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count >= len(self.entries):
            return [entry.name for entry in self.entries]
        indexes = self.rng.choice(len(self.entries), size=count, replace=False)
        return [self.entries[int(index)].name for index in indexes]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"version": 1, "entries": [asdict(entry) for entry in self.entries]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, seed: int) -> "LeagueRoster":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") != 1 or not isinstance(value.get("entries"), list):
            raise ValueError("unsupported league roster")
        roster = cls(seed)
        for entry in value["entries"]:
            roster.add(
                str(entry["name"]),
                str(entry["checkpoint"]),
                frozen=bool(entry["frozen"]),
            )
        return roster


def sample_public_belief(env: PropertyTradingEnv, seed: int) -> PropertyTradingEnv:
    """Clone public state while resampling the order of every hidden deck."""
    simulation = copy.deepcopy(env)
    simulation.rng = np.random.default_rng(seed)
    for deck in (simulation.chance_deck, simulation.community_deck):
        deck.sort()
        simulation.rng.shuffle(deck)
    return simulation


class SearchTeacher:
    """Small seeded rollout teacher that never observes future random events."""

    def __init__(self, seed: int, simulations: int = 64, depth: int = 48):
        if simulations < 1 or depth < 1:
            raise ValueError("simulations and depth must be positive")
        self.rng = np.random.default_rng(seed)
        self.simulations = simulations
        self.depth = depth

    @staticmethod
    def _public_score(env: PropertyTradingEnv, seat: int) -> float:
        if env.done:
            return float(env.terminal_rewards[seat])
        living = np.flatnonzero(env.alive)
        if seat not in living:
            return -1.0
        rivals = [int(index) for index in living if int(index) != seat]
        rival_cash = float(np.mean(env.cash[rivals])) if rivals else 0.0
        owned = float(np.count_nonzero(env.owner == seat))
        rival_owned = float(
            np.mean([np.count_nonzero(env.owner == rival) for rival in rivals]),
        ) if rivals else 0.0
        return float(np.tanh(((float(env.cash[seat]) - rival_cash) / 1500) + (owned - rival_owned) / 10))

    def _rollout(self, env: PropertyTradingEnv, action: Action, seat: int) -> float:
        simulation = sample_public_belief(
            env,
            int(self.rng.integers(1, np.iinfo(np.int32).max)),
        )
        try:
            simulation.step(action)
        except (RuntimeError, ValueError):
            return -1.0
        for _ in range(self.depth - 1):
            if simulation.done:
                break
            actions = simulation.legal_actions()
            if not actions:
                break
            choice = actions[int(self.rng.integers(len(actions)))]
            simulation.step(choice, validate=False)
        return self._public_score(simulation, seat)

    def select_action(self, env: PropertyTradingEnv) -> tuple[Action, np.ndarray]:
        view = public_view(env, env.actor)
        actions = view["legal_actions"]
        if not actions:
            raise RuntimeError("no legal action is available")
        totals = np.zeros(len(actions), dtype=np.float64)
        visits = np.zeros(len(actions), dtype=np.int32)
        for index in range(max(self.simulations, len(actions))):
            action_index = index % len(actions)
            totals[action_index] += self._rollout(env, actions[action_index], env.actor)
            visits[action_index] += 1
        means = totals / np.maximum(visits, 1)
        shifted = means - means.max()
        policy = np.exp(shifted)
        policy /= policy.sum()
        best = int(np.flatnonzero(means == means.max())[0])
        return actions[best], policy.astype(np.float32)
