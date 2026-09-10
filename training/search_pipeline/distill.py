from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from training.zero_knowledge.encoding import collate_actions, collate_observations
from training.zero_knowledge.env import Action, ActionKind
from training.zero_knowledge.model import ModelConfig, PropertyTradingPolicy

from .dataset import read_teacher_dataset
from .config import load_search_config


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
    parser = argparse.ArgumentParser(description="Distill a fast policy from search-teacher data")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
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
    samples = read_teacher_dataset(args.dataset)
    model, metrics = _train(
        samples,
        config=config,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
    )
    pipeline_config = load_search_config()
    teacher_versions = sorted({str(sample.get("teacher_version", "unknown")) for sample in samples})
    teacher_configs = {
        json.dumps(sample.get("teacher_config", {}), sort_keys=True)
        for sample in samples
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({
        "version": 1,
        "stage": "search-teacher-distillation",
        "policy_version": pipeline_config["student"]["version"],
        "training_seed": args.seed,
        "training_steps": args.steps,
        "dataset_path": args.dataset.as_posix(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "teacher_versions": teacher_versions,
        "teacher_configs": [json.loads(value) for value in sorted(teacher_configs)],
        "model_config": asdict(config),
        "model": model.state_dict(),
        "metrics": metrics,
        "samples": len(samples),
    }, temporary)
    temporary.replace(args.output)
    print({"output": str(args.output), "samples": len(samples), **metrics})


if __name__ == "__main__":
    main()
