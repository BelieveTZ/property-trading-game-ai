from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

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
) -> dict[str, object]:
    if games < 1 or not player_counts:
        raise ValueError("games and player_counts must be non-empty")
    if not roster.entries or any(not entry.frozen for entry in roster.entries):
        raise ValueError("league training requires frozen opponents")
    target = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
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
    for game_index in range(games):
        player_count = player_counts[game_index % len(player_counts)]
        game_seed = seed + game_index
        player_count_game = games_by_player_count[player_count]
        candidate_seat = player_count_game % player_count
        opponent_rotation = (player_count_game // player_count) % len(roster.entries)
        games_by_player_count[player_count] += 1
        candidate_seats.append(candidate_seat)
        env = PropertyTradingEnv(player_count, seed=game_seed, max_rounds=max_rounds)
        runtimes = [SeatRuntime.create(game_seed, seat) for seat in range(player_count)]
        seat_seeds.append([runtime.seed for runtime in runtimes])
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
        opponent_seats.append(seat_policies)
        game_samples: list[dict[str, object]] = []
        for _ in range(max_rounds * player_count * 300):
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
        seat_rewards.append([runtime.reward for runtime in runtimes])
        reward = runtimes[candidate_seat].reward
        for sample in game_samples:
            sample["outcome"] = reward
        all_samples.extend(game_samples)
        rewards.append(reward)
        decisions += len(game_samples)
    losses = _update_candidate(candidate, all_samples, target)
    config = load_search_config()
    metrics: dict[str, object] = {
        "games": games,
        "decisions": decisions,
        "averageReward": float(np.mean(rewards)),
        "seatSeeds": seat_seeds,
        "seatRewards": seat_rewards,
        "candidateSeats": candidate_seats,
        "seatPolicies": opponent_seats,
        **losses,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({
        "version": 1,
        "stage": "league-self-play",
        "policy_version": config["student"]["version"],
        "model_config": source_payload["model_config"],
        "model": candidate.state_dict(),
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "league": [asdict(entry) for entry in roster.entries],
        "metrics": metrics,
    }, temporary)
    temporary.replace(output)
    return metrics
