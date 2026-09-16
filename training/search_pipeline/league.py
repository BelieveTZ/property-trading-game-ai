from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from training.zero_knowledge.encoding import collate_actions, collate_observations
from training.zero_knowledge.env import Action, PropertyTradingEnv
from training.zero_knowledge.model import ModelConfig, PropertyTradingPolicy

from .config import load_search_config
from .protocol import LeagueRoster, SeatRuntime, public_view
from .policies import load_frozen_policy


def _load_model(path: Path, device: torch.device) -> tuple[PropertyTradingPolicy, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PropertyTradingPolicy(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def _sample_action(
    model: PropertyTradingPolicy,
    view: dict[str, object],
    runtime: SeatRuntime,
    device: torch.device,
) -> tuple[Action, int]:
    actions = view["legal_actions"]
    observation = collate_observations([view["observation"]], device)
    action_data, mask = collate_actions(
        [actions],
        [int(view["actor"])],
        [int(view["player_count"])],
        device,
    )
    hidden = runtime.memory.get("hidden")
    with torch.no_grad():
        logits, _, next_hidden = model(observation, action_data, mask, hidden)
        probabilities = torch.softmax(logits[0].float(), dim=-1).cpu().numpy()
    index = int(runtime.rng.choice(len(actions), p=probabilities))
    runtime.memory["hidden"] = next_hidden.detach()
    return actions[index], index


def _update_candidate(
    model: PropertyTradingPolicy,
    samples: list[dict[str, object]],
    device: torch.device,
) -> dict[str, float]:
    if not samples:
        return {"policyLoss": 0.0, "valueLoss": 0.0}
    observations = [sample["observation"] for sample in samples]
    actions = [sample["legal_actions"] for sample in samples]
    actors = [int(sample["actor"]) for sample in samples]
    player_counts = [int(sample["player_count"]) for sample in samples]
    selected = torch.as_tensor(
        [int(sample["selected_action"]) for sample in samples],
        dtype=torch.int64,
        device=device,
    )
    outcomes = torch.as_tensor(
        [float(sample["outcome"]) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    observation_batch = collate_observations(observations, device)
    action_batch, mask = collate_actions(actions, actors, player_counts, device)
    hidden = torch.stack([
        torch.as_tensor(sample["hidden"], dtype=torch.float32, device=device).reshape(-1)
        for sample in samples
    ])
    model.train()
    logits, values, _ = model(observation_batch, action_batch, mask, hidden)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    chosen = log_probabilities.gather(1, selected.unsqueeze(1)).squeeze(1)
    advantage = outcomes - values.float().detach()
    policy_loss = -(advantage * chosen).mean()
    value_loss = 0.5 * (values.float() - outcomes).square().mean()
    loss = policy_loss + value_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "policyLoss": float(policy_loss.detach().item()),
        "valueLoss": float(value_loss.detach().item()),
    }


def run_league_training(
    *,
    checkpoint: Path,
    roster: LeagueRoster,
    output: Path,
    games: int,
    player_counts: tuple[int, ...],
    seed: int,
    max_rounds: int,
    device: str,
    run_directory: Path | None = None,
    should_pause: Callable[[int, int], bool] | None = None,
    gpu_hours: float | None = None,
) -> dict[str, object]:
    if games < 1 or not player_counts:
        raise ValueError("games and player_counts must be non-empty")
    if gpu_hours is not None and gpu_hours <= 0:
        raise ValueError("gpu_hours must be positive")
    if not roster.entries or any(not entry.frozen for entry in roster.entries):
        raise ValueError("league training requires frozen opponents")
    target = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    config = load_search_config()
    update_every_games = int(config["league"].get("updateEveryGames", 24))
    if update_every_games < 1:
        raise ValueError("league updateEveryGames must be positive")
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    signature = {
        "source_checkpoint_sha256": source_hash,
        "league": [asdict(entry) for entry in roster.entries],
        "games": games,
        "player_counts": list(player_counts),
        "seed": seed,
        "max_rounds": max_rounds,
        "gpu_hours": gpu_hours,
        "update_every_games": update_every_games,
    }
    checkpoint_path = (
        run_directory / "league-checkpoint.pt" if run_directory is not None else None
    )
    state_path = run_directory / "league-state.json" if run_directory is not None else None

    def write_state(status: str, completed_games: int, gpu_seconds: float = 0.0) -> None:
        if state_path is None:
            return
        run_directory.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "stage": "league-self-play",
                    "status": status,
                    "completed_games": completed_games,
                    "total_games": games,
                    **({"gpu_seconds": gpu_seconds} if gpu_hours is not None else {}),
                    **signature,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(state_path)

    candidate, source_payload = _load_model(checkpoint, target)
    opponent_models = {
        entry.name: (
            load_frozen_policy(Path(entry.checkpoint), device=str(target))
            if Path(entry.checkpoint).suffix.lower() == ".json"
            else _load_model(Path(entry.checkpoint), target)[0].eval()
        )
        for entry in roster.entries
    }
    all_samples: list[dict[str, object]] = []
    rewards: list[float] = []
    seat_seeds: list[list[int]] = []
    seat_rewards: list[list[float]] = []
    opponent_seats: list[list[str]] = []
    candidate_seats: list[int] = []
    games_by_player_count = {player_count: 0 for player_count in player_counts}
    decisions = 0
    start_game = 0
    accumulated_gpu_seconds = 0.0
    updates = 0
    last_losses = {"policyLoss": 0.0, "valueLoss": 0.0}
    if checkpoint_path is not None and checkpoint_path.is_file():
        progress = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if progress.get("version") != 1 or progress.get("signature") != signature:
            raise ValueError("league checkpoint does not match requested configuration")
        start_game = int(progress["next_game_index"])
        all_samples = list(progress["all_samples"])
        rewards = list(progress["rewards"])
        seat_seeds = list(progress["seat_seeds"])
        seat_rewards = list(progress["seat_rewards"])
        opponent_seats = list(progress["opponent_seats"])
        candidate_seats = list(progress["candidate_seats"])
        games_by_player_count = dict(progress["games_by_player_count"])
        decisions = int(progress["decisions"])
        accumulated_gpu_seconds = float(progress.get("gpu_seconds", 0.0))
        updates = int(progress.get("updates", 0))
        last_losses = dict(progress.get("last_losses", last_losses))
        if "candidate_model" in progress:
            candidate.load_state_dict(progress["candidate_model"])

    segment_started = time.perf_counter()

    def current_gpu_seconds() -> float:
        return accumulated_gpu_seconds + (time.perf_counter() - segment_started)

    def save_checkpoint(game_index: int) -> float:
        if checkpoint_path is None:
            raise RuntimeError("checkpointing league training requires a run directory")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        gpu_seconds = current_gpu_seconds()
        torch.save(
            {
                "version": 1,
                "stage": "league-self-play-checkpoint",
                "signature": signature,
                "next_game_index": game_index,
                "all_samples": all_samples,
                "rewards": rewards,
                "seat_seeds": seat_seeds,
                "seat_rewards": seat_rewards,
                "opponent_seats": opponent_seats,
                "candidate_seats": candidate_seats,
                "games_by_player_count": games_by_player_count,
                "decisions": decisions,
                "gpu_seconds": gpu_seconds,
                "candidate_model": candidate.state_dict(),
                "updates": updates,
                "last_losses": last_losses,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        return gpu_seconds

    def pause(game_index: int) -> dict[str, object]:
        gpu_seconds = save_checkpoint(game_index)
        write_state("paused", game_index, gpu_seconds)
        result = {
            "status": "paused",
            "completedGames": game_index,
            "games": games,
        }
        if gpu_hours is not None:
            result["gpuSeconds"] = gpu_seconds
        return result

    write_state("running", start_game, accumulated_gpu_seconds)
    budget_reached = False

    def update_pending_samples() -> None:
        nonlocal updates, last_losses
        if not all_samples:
            return
        last_losses = _update_candidate(candidate, all_samples, target)
        all_samples.clear()
        updates += 1

    for game_index in range(start_game, games):
        if should_pause is not None and should_pause(game_index, 0):
            return pause(game_index)
        player_count = player_counts[game_index % len(player_counts)]
        game_seed = seed + game_index
        player_count_game = games_by_player_count[player_count]
        candidate_seat = player_count_game % player_count
        opponent_rotation = (player_count_game // player_count) % len(roster.entries)
        env = PropertyTradingEnv(player_count, seed=game_seed, max_rounds=max_rounds)
        runtimes = [SeatRuntime.create(game_seed, seat) for seat in range(player_count)]
        game_seat_seeds = [runtime.seed for runtime in runtimes]
        sampled_names = roster.sample(min(len(roster.entries), player_count - 1))
        opponent_names = [
            sampled_names[(opponent_rotation + index) % len(sampled_names)]
            for index in range(player_count - 1)
        ]
        seat_policies: list[str] = []
        opponent_index = 0
        for seat in range(player_count):
            if seat == candidate_seat:
                seat_policies.append("candidate")
            else:
                seat_policies.append(opponent_names[opponent_index])
                opponent_index += 1
        game_samples: list[dict[str, object]] = []
        for decision_index in range(max_rounds * player_count * 300):
            if should_pause is not None and should_pause(game_index, decision_index):
                return pause(game_index)
            if env.done:
                break
            seat = env.actor
            view = public_view(env, seat)
            model = candidate if seat == candidate_seat else opponent_models[seat_policies[seat]]
            hidden_before = runtimes[seat].memory.get("hidden")
            if isinstance(model, PropertyTradingPolicy):
                action, selected = _sample_action(model, view, runtimes[seat], target)
            else:
                action = model(view)
                selected = view["legal_actions"].index(action)
            if seat == candidate_seat:
                game_samples.append({
                    "observation": view["observation"],
                    "legal_actions": view["legal_actions"],
                    "actor": view["actor"],
                    "player_count": view["player_count"],
                    "selected_action": selected,
                    "hidden": (
                        hidden_before.squeeze(0).detach().cpu()
                        if isinstance(hidden_before, torch.Tensor)
                        else torch.zeros(candidate.hidden_size)
                    ),
                })
            env.step(action)
        if not env.done:
            raise RuntimeError("league game exceeded its deterministic decision budget")
        for seat, runtime in enumerate(runtimes):
            runtime.reward = float(env.terminal_rewards[seat])
        games_by_player_count[player_count] += 1
        candidate_seats.append(candidate_seat)
        seat_seeds.append(game_seat_seeds)
        opponent_seats.append(seat_policies)
        seat_rewards.append([runtime.reward for runtime in runtimes])
        reward = runtimes[candidate_seat].reward
        for sample in game_samples:
            sample["outcome"] = reward
        all_samples.extend(game_samples)
        rewards.append(reward)
        decisions += len(game_samples)
        completed_games = game_index + 1
        if completed_games % update_every_games == 0:
            update_pending_samples()
            if checkpoint_path is not None:
                save_checkpoint(completed_games)
        write_state("running", completed_games, current_gpu_seconds())
        if gpu_hours is not None and current_gpu_seconds() >= gpu_hours * 3600:
            budget_reached = True
            break
    update_pending_samples()
    completed_games = len(rewards)
    gpu_seconds = current_gpu_seconds()
    metrics: dict[str, object] = {
        "games": completed_games,
        "decisions": decisions,
        "averageReward": float(np.mean(rewards)),
        "seatSeeds": seat_seeds,
        "seatRewards": seat_rewards,
        "candidateSeats": candidate_seats,
        "seatPolicies": opponent_seats,
        "updates": updates,
        **last_losses,
    }
    if gpu_hours is not None:
        metrics.update({
            "status": "budget-reached" if budget_reached else "completed",
            "completedGames": completed_games,
            "gpuSeconds": gpu_seconds,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({
        "version": 1,
        "stage": "league-self-play",
        "policy_version": config["student"]["version"],
        "model_config": source_payload["model_config"],
        "model": candidate.state_dict(),
        "source_checkpoint_sha256": source_hash,
        "league": [asdict(entry) for entry in roster.entries],
        "metrics": metrics,
    }, temporary)
    temporary.replace(output)
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    write_state("budget-reached" if budget_reached else "completed", completed_games, gpu_seconds)
    return metrics
