from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True, eq=True)
class SearchRunState:
    version: int = 1
    stage: str = "teacher-data"
    completed_games: int = 0
    gpu_seconds: float = 0.0
    completed_milestones: list[int] = field(default_factory=list)


class SearchRunControl:
    def __init__(self, run_directory: Path):
        self.run_directory = run_directory
        self.state_path = run_directory / "search-run.json"
        self.pause_path = run_directory / "pause.request"

    def save(self, state: SearchRunState) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def load(self) -> SearchRunState:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.get("version") != 1:
            raise ValueError("unsupported search run version")
        return SearchRunState(**value)

    def begin(self) -> SearchRunState:
        """Load resumable state and consume a pause request from the prior run."""
        state = self.load() if self.state_path.exists() else SearchRunState()
        self.clear_pause()
        return state

    def request_pause(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.pause_path.touch()

    def pause_requested(self) -> bool:
        return self.pause_path.exists()

    def clear_pause(self) -> None:
        self.pause_path.unlink(missing_ok=True)
