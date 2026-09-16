"""Reproducible public-information search-teacher training primitives."""

from .dataset import (
    TeacherEpisodePaused,
    generate_teacher_episode,
    generate_teacher_episode_file,
    read_teacher_dataset,
    write_teacher_dataset,
)
from .control import SearchRunControl, SearchRunState
from .config import load_search_config
from .evaluation import (
    EvaluationGame,
    build_balanced_evaluation_schedule,
    evaluate_policy_league,
    freeze_due_milestones,
)
from .protocol import (
    EvaluationSchedule,
    LeagueRoster,
    SearchTeacher,
    SeatRuntime,
    public_view,
    sample_public_belief,
)
from .league import run_league_training
from .upper_bound import ConfiguredSearchPolicy
from .policies import load_frozen_policy

__all__ = [
    "EvaluationSchedule",
    "EvaluationGame",
    "LeagueRoster",
    "ConfiguredSearchPolicy",
    "SearchTeacher",
    "SearchRunControl",
    "SearchRunState",
    "sample_public_belief",
    "SeatRuntime",
    "TeacherEpisodePaused",
    "generate_teacher_episode",
    "generate_teacher_episode_file",
    "generate_teacher_games",
    "build_balanced_evaluation_schedule",
    "evaluate_policy_league",
    "freeze_due_milestones",
    "load_search_config",
    "load_frozen_policy",
    "public_view",
    "read_teacher_dataset",
    "run_league_training",
    "teacher_generation_defaults",
    "write_teacher_dataset",
]


def __getattr__(name: str):
    if name in {"generate_teacher_games", "teacher_generation_defaults"}:
        from .generate import generate_teacher_games, teacher_generation_defaults

        return {
            "generate_teacher_games": generate_teacher_games,
            "teacher_generation_defaults": teacher_generation_defaults,
        }[name]
    raise AttributeError(name)
