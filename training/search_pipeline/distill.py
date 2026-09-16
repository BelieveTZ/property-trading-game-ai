from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from training.zero_knowledge.encoding import collate_actions, collate_observations
from training.zero_knowledge.env import Action, ActionKind
from training.zero_knowledge.model import ModelConfig, PropertyTradingPolicy

from .dataset import read_teacher_dataset
from .config import load_search_config
from .control import record_training_command


SAMPLING_VERSION = "seeded-with-replacement-v1"
OPTIMIZER_CONFIG = {
    "name": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
}


def distillation_defaults() -> dict[str, int]:
    distillation = load_search_config()["student"]["distillation"]
    return {
        "steps": int(distillation["steps"]),
        "batch_size": int(distillation["batchSize"]),
        "checkpoint_every": int(distillation["checkpointEvery"]),
    }


class IndexedTeacherDataset:
    """Random-access JSONL samples without retaining the dataset in memory."""

    def __init__(self, path: Path, run_directory: Path):
        self.path = path
        self.run_directory = run_directory
        self.index_path = run_directory / "distill-offsets.npy"
        self.manifest_path = run_directory / "distill-dataset.json"
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.offsets, self.sha256 = self._load_or_build_index()

    def _load_or_build_index(self) -> tuple[np.ndarray, str]:
        stat = self.path.stat()
        if self.manifest_path.is_file() and self.index_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("version") == 1
                and manifest.get("dataset_path") == str(self.path.resolve())
                and manifest.get("dataset_size") == stat.st_size
                and manifest.get("dataset_mtime_ns") == stat.st_mtime_ns
            ):
                offsets = np.load(self.index_path, mmap_mode="r")
                if len(offsets) == int(manifest["samples"]):
                    return offsets, str(manifest["dataset_sha256"])

        offsets: list[int] = []
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if line.strip():
                    offsets.append(offset)
        if not offsets:
            raise ValueError("teacher dataset must be non-empty")

        temporary_index = self.index_path.with_suffix(".tmp.npy")
        np.save(temporary_index, np.asarray(offsets, dtype=np.uint64))
        temporary_index.replace(self.index_path)
        manifest = {
            "version": 1,
            "dataset_path": str(self.path.resolve()),
            "dataset_size": stat.st_size,
            "dataset_mtime_ns": stat.st_mtime_ns,
            "dataset_sha256": digest.hexdigest(),
            "samples": len(offsets),
        }
        temporary_manifest = self.manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_manifest.replace(self.manifest_path)
        return np.load(self.index_path, mmap_mode="r"), digest.hexdigest()

    def __len__(self) -> int:
        return len(self.offsets)

    def read(self, indices: np.ndarray) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        with self.path.open("rb") as stream:
            for index in indices:
                stream.seek(int(self.offsets[int(index)]))
                samples.append(json.loads(stream.readline()))
        return samples


def _action(value: dict[str, int]) -> Action:
    return Action(
        kind=ActionKind(int(value["kind"])),
        tile=int(value.get("tile", -1)),
        target=int(value.get("target", -1)),
        cash=int(value.get("cash", 0)),
        give_tile=int(value.get("give_tile", -1)),
        take_tile=int(value.get("take_tile", -1)),
        give_card=int(value.get("give_card", 0)),
        take_card=int(value.get("take_card", 0)),
    )


def _train(
    samples: list[dict[str, object]],
    *,
    config: ModelConfig,
    steps: int,
    device: str,
    seed: int,
) -> tuple[PropertyTradingPolicy, dict[str, float]]:
    if not samples or steps < 1:
        raise ValueError("samples and steps must be non-empty")
    target_device = torch.device(device)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PropertyTradingPolicy(config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(OPTIMIZER_CONFIG["learning_rate"]),
        weight_decay=float(OPTIMIZER_CONFIG["weight_decay"]),
    )
    observations = [
        {name: np.asarray(value, dtype=np.float32) for name, value in sample["observation"].items()}
        for sample in samples
    ]
    action_lists = [[_action(value) for value in sample["legal_actions"]] for sample in samples]
    actors = [int(sample["actor"]) for sample in samples]
    player_counts = [int(sample["player_count"]) for sample in samples]
    observation_batch = collate_observations(observations, target_device)
    action_batch, mask = collate_actions(action_lists, actors, player_counts, target_device)
    teacher = torch.zeros(mask.shape, dtype=torch.float32, device=target_device)
    for row, sample in enumerate(samples):
        values = torch.as_tensor(sample["policy"], dtype=torch.float32, device=target_device)
        teacher[row, : values.shape[0]] = values
    outcomes = torch.as_tensor(
        [float(sample["outcome"]) for sample in samples],
        dtype=torch.float32,
        device=target_device,
    )
    metrics = {"policy_loss": 0.0, "value_loss": 0.0}
    model.train()
    for _ in range(steps):
        logits, values, _ = model(observation_batch, action_batch, mask)
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        policy_loss = -(teacher * log_probabilities).sum(dim=-1).mean()
        value_loss = 0.5 * (values.float() - outcomes).square().mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
        }
    return model, metrics


def _train_step(
    model: PropertyTradingPolicy,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, object]],
    *,
    device: torch.device,
) -> dict[str, float]:
    observations = [
        {name: np.asarray(value, dtype=np.float32) for name, value in sample["observation"].items()}
        for sample in samples
    ]
    action_lists = [[_action(value) for value in sample["legal_actions"]] for sample in samples]
    actors = [int(sample["actor"]) for sample in samples]
    player_counts = [int(sample["player_count"]) for sample in samples]
    observation_batch = collate_observations(observations, device)
    action_batch, mask = collate_actions(action_lists, actors, player_counts, device)
    teacher = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    for row, sample in enumerate(samples):
        values = torch.as_tensor(sample["policy"], dtype=torch.float32, device=device)
        teacher[row, : values.shape[0]] = values
    outcomes = torch.as_tensor(
        [float(sample["outcome"]) for sample in samples],
        dtype=torch.float32,
        device=device,
    )

    logits, values, _ = model(observation_batch, action_batch, mask)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    policy_loss = -(teacher * log_probabilities).sum(dim=-1).mean()
    value_loss = 0.5 * (values.float() - outcomes).square().mean()
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
    }


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_torch_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_distillation(
    *,
    dataset_path: Path,
    output: Path,
    run_directory: Path,
    config: ModelConfig,
    steps: int,
    batch_size: int,
    checkpoint_every: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    if steps < 1 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("steps, batch size, and checkpoint interval must be positive")
    target_device = torch.device(device)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    indexed = IndexedTeacherDataset(dataset_path, run_directory)
    model = PropertyTradingPolicy(config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(OPTIMIZER_CONFIG["learning_rate"]),
        weight_decay=float(OPTIMIZER_CONFIG["weight_decay"]),
    )
    model.train()
    checkpoint_path = run_directory / "distill-checkpoint.pt"
    state_path = run_directory / "distill-state.json"
    pause_path = run_directory / "distill-pause.request"
    metrics = {"policy_loss": 0.0, "value_loss": 0.0}
    gpu_seconds = 0.0
    completed_steps = 0
    signature = {
        "dataset_sha256": indexed.sha256,
        "model_config": asdict(config),
        "training_seed": seed,
        "training_batch_size": batch_size,
        "sampling_version": SAMPLING_VERSION,
        "optimizer_config": OPTIMIZER_CONFIG,
    }
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
        if saved.get("version") != 1 or any(
            saved.get(name) != value for name, value in signature.items()
        ):
            raise ValueError("distillation checkpoint does not match requested configuration")
        completed_steps = int(saved["completed_steps"])
        if completed_steps > steps:
            raise ValueError("distillation checkpoint is ahead of requested steps")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        metrics = {
            "policy_loss": float(saved["metrics"]["policy_loss"]),
            "value_loss": float(saved["metrics"]["value_loss"]),
        }
        gpu_seconds = float(saved.get("gpu_seconds", 0.0))
    resumed_from_step = completed_steps

    def state(status: str, current_steps: int) -> dict[str, object]:
        return {
            "version": 1,
            "stage": "search-teacher-distillation",
            "status": status,
            "completed_steps": current_steps,
            "total_steps": steps,
            "resumed_from_step": resumed_from_step,
            "batch_size": batch_size,
            "samples": len(indexed),
            "dataset_sha256": indexed.sha256,
            "device": str(target_device),
            "gpu_seconds": gpu_seconds,
            "metrics": metrics,
        }

    _write_json_atomic(state_path, state("running", completed_steps))
    for step in range(completed_steps, steps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        indices = rng.integers(0, len(indexed), size=batch_size)
        samples = indexed.read(indices)
        started = time.perf_counter()
        metrics = _train_step(model, optimizer, samples, device=target_device)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
            gpu_seconds += time.perf_counter() - started
        completed_steps = step + 1
        pause_requested = pause_path.exists()
        if (
            completed_steps % checkpoint_every == 0
            or completed_steps == steps
            or pause_requested
        ):
            checkpoint = {
                "version": 1,
                "stage": "search-teacher-distillation-checkpoint",
                **signature,
                "completed_steps": completed_steps,
                "gpu_seconds": gpu_seconds,
                "metrics": metrics,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            _save_torch_atomic(checkpoint_path, checkpoint)
            _write_json_atomic(
                state_path,
                state("paused" if pause_requested else "running", completed_steps),
            )
        if pause_requested:
            return {
                "status": "paused",
                "samples": len(indexed),
                "completed_steps": completed_steps,
                "metrics": metrics,
            }

    pipeline_config = load_search_config()
    teacher_versions: set[str] = set()
    teacher_configs: set[str] = set()
    for start in range(0, len(indexed), 1024):
        stop = min(start + 1024, len(indexed))
        for sample in indexed.read(np.arange(start, stop, dtype=np.int64)):
            teacher_versions.add(str(sample.get("teacher_version", "unknown")))
            teacher_configs.add(json.dumps(sample.get("teacher_config", {}), sort_keys=True))
    payload = {
        "version": 1,
        "stage": "search-teacher-distillation",
        "policy_version": pipeline_config["student"]["version"],
        "training_seed": seed,
        "training_steps": steps,
        "training_batch_size": batch_size,
        "sampling_version": SAMPLING_VERSION,
        "optimizer": OPTIMIZER_CONFIG,
        "dataset_path": dataset_path.as_posix(),
        "dataset_sha256": indexed.sha256,
        "teacher_versions": sorted(teacher_versions),
        "teacher_configs": [json.loads(value) for value in sorted(teacher_configs)],
        "model_config": asdict(config),
        "model": model.state_dict(),
        "metrics": metrics,
        "samples": len(indexed),
        "gpu_seconds": gpu_seconds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_torch_atomic(output, payload)
    _write_json_atomic(state_path, state("completed", steps))
    return payload


def distill_batch(
    samples: list[dict[str, object]],
    *,
    config: ModelConfig,
    steps: int,
    device: str,
    seed: int = 20260910,
) -> dict[str, float]:
    _, metrics = _train(samples, config=config, steps=steps, device=device, seed=seed)
    return metrics


def parse_args() -> argparse.Namespace:
    pipeline_config = load_search_config()
    defaults = distillation_defaults()
    parser = argparse.ArgumentParser(description="Distill a fast policy from search-teacher data")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=int, default=defaults["steps"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=defaults["checkpoint_every"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=512)
    parser.add_argument("--seed", type=int, default=int(pipeline_config["student"]["seed"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        feedforward=args.feedforward,
    )
    run_directory = args.run_dir or args.output.parent
    run_directory.mkdir(parents=True, exist_ok=True)
    record_training_command(
        run_directory,
        module="training.search_pipeline.distill",
        arguments=sys.argv[1:],
        pid_file="distill-pid.txt",
        pause_file="distill-pause.request",
        state_file="distill-state.json",
        stdout_log="distill.stdout.log",
        stderr_log="distill.stderr.log",
    )
    (run_directory / "distill-pause.request").unlink(missing_ok=True)
    (run_directory / "distill-pid.txt").write_text(str(os.getpid()), encoding="ascii")
    payload = run_distillation(
        dataset_path=args.dataset,
        output=args.output,
        run_directory=run_directory,
        config=config,
        steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        seed=args.seed,
    )
    if payload.get("status") == "paused":
        print({
            "status": "paused",
            "completed_steps": payload["completed_steps"],
            "samples": payload["samples"],
        })
    else:
        print({
            "output": str(args.output),
            "samples": payload["samples"],
            **payload["metrics"],
        })


if __name__ == "__main__":
    main()
