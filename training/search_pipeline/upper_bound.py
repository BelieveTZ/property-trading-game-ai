from __future__ import annotations

from pathlib import Path

from training.zero_knowledge.env import Action, PropertyTradingEnv

from .config import DEFAULT_CONFIG_PATH, load_search_config
from .protocol import SearchTeacher


class ConfiguredSearchPolicy:
    """Versioned high-budget policy used for upper-bound evaluation."""

    def __init__(self, *, seed: int, config_path: Path = DEFAULT_CONFIG_PATH):
        self.seed = seed
        self.config_path = config_path
        config = load_search_config(config_path)
        self.config_version = int(config["version"])
        policy = config["policies"]["upperBound"]
        self.version = str(policy["version"])
        self.simulations = int(policy["simulations"])
        self.depth = int(policy["depth"])
        self.teacher = self._teacher(seed)

    def _teacher(self, seed: int) -> SearchTeacher:
        return SearchTeacher(seed=seed, simulations=self.simulations, depth=self.depth)

    def start_game(self, assignment: object) -> None:
        game_seed = int(getattr(assignment, "seed"))
        candidate_seat = int(getattr(assignment, "candidate_seat"))
        self.teacher = self._teacher(self.seed + game_seed * 7 + candidate_seat)

    def select_action(self, env: PropertyTradingEnv) -> Action:
        action, _ = self.teacher.select_action(env)
        return action

    def provenance(self) -> dict[str, object]:
        return {
            "policyVersion": self.version,
            "teacherSeed": self.seed,
            "simulations": self.simulations,
            "depth": self.depth,
            "configVersion": self.config_version,
            "configPath": str(self.config_path.resolve()),
        }
