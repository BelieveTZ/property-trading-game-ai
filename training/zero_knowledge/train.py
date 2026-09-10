from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from .env import Action, MAX_PLAYERS, MonopolyEnv
from .encoding import collate_action_feature_rows, collate_actions, collate_observations
from .model import ModelConfig, MonopolyPolicy


@dataclass(slots=True)
class Transition:
    observation: dict[str, np.ndarray]
    action_features: dict[str, np.ndarray]
    action_index: int
    old_log_prob: float
    old_value: float
    hidden: np.ndarray
    outcome: float = 0.0


@dataclass(slots=True)
class TrainState:
    update: int = 0
    games: int = 0
    decisions: int = 0
    best_score: float = -1e9
    no_improvement: int = 0
    league_window_wins: int = 0
    league_window_total: int = 0
    started_at: float = 0.0


STOP_REQUESTED = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def gpu_temperature() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


class LeagueTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.league_dir = self.run_dir / "league"
        self.league_dir.mkdir(exist_ok=True)
        self.pause_path = self.run_dir / "pause.request"
        self.status_path = self.run_dir / "status.json"
        self.checkpoint_path = self.run_dir / "checkpoint.pt"
        self.log_path = self.run_dir / "metrics.jsonl"
        selected_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
        self.device = torch.device(selected_device)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
        self.rng = np.random.default_rng(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        config = ModelConfig(
            d_model=args.d_model,
            nhead=args.heads,
            layers=args.layers,
            feedforward=args.feedforward,
            dropout=args.dropout,
        )
        self.model = MonopolyPolicy(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=max(4, args.patience // 3), min_lr=1e-5
        )
        self.state = TrainState(started_at=time.time())
        self.snapshots: list[MonopolyPolicy] = []
        self.snapshot_updates: list[int] = []
        self.autocast_dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        if args.resume and self.checkpoint_path.exists():
            self._load_checkpoint()
        elif args.init_from:
            source = torch.load(Path(args.init_from).resolve(), map_location=self.device, weights_only=False)
            self.model.load_state_dict(source["model"])
        self._load_league()

    def _write_status(self, state: str, **extra: object) -> None:
        payload = {
            "state": state,
            "pid": os.getpid(),
            "device": str(self.device),
            "update": self.state.update,
            "games": self.state.games,
            "decisions": self.state.decisions,
            "best_score": self.state.best_score,
            "league_size": len(self.snapshots),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **extra,
        }
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.status_path)

    def _save_checkpoint(self, reason: str) -> None:
        checkpoint = {
            "version": 1,
            "reason": reason,
            "args": vars(self.args),
            "model_config": asdict(self.model.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_state": asdict(self.state),
            "numpy_rng": self.rng.bit_generator.state,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.checkpoint_path)

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state = TrainState(**checkpoint["train_state"])
        self.rng.bit_generator.state = checkpoint["numpy_rng"]
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])

    def _load_league(self) -> None:
        all_paths = sorted(self.league_dir.glob("snapshot-*.pt"))
        paths = all_paths[-self.args.max_snapshots :]
        if all_paths and all_paths[0] not in paths and self.args.max_snapshots > 1:
            paths = [all_paths[0], *paths[-(self.args.max_snapshots - 1) :]]
        for path in paths:
            payload = torch.load(path, map_location=self.device, weights_only=True)
            snapshot = MonopolyPolicy(self.model.config).to(self.device)
            snapshot.load_state_dict(payload["model"])
            snapshot.eval()
            snapshot.requires_grad_(False)
            self.snapshots.append(snapshot)
            self.snapshot_updates.append(int(payload["update"]))

    def _add_snapshot(self) -> None:
        update = self.state.update
        path = self.league_dir / f"snapshot-{update:07d}.pt"
        torch.save({"update": update, "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()}}, path)
        snapshot = copy.deepcopy(self.model).eval()
        snapshot.requires_grad_(False)
        self.snapshots.append(snapshot)
        self.snapshot_updates.append(update)
        if len(self.snapshots) > self.args.max_snapshots:
            remove_at = 1 if len(self.snapshots) > 1 else 0
            self.snapshots.pop(remove_at)
            self.snapshot_updates.pop(remove_at)
        paths = sorted(self.league_dir.glob("snapshot-*.pt"))
        while len(paths) > self.args.max_snapshot_files:
            paths[1].unlink(missing_ok=True)
            paths = sorted(self.league_dir.glob("snapshot-*.pt"))

    def _player_count(self) -> int:
        choices = [int(value) for value in self.args.players.split(",")]
        return int(self.rng.choice(choices))

    def _seat_policies(self, player_count: int, game_index: int) -> list[int]:
        learner = game_index % player_count
        if not self.snapshots or self.rng.random() < self.args.random_opponent_fraction:
            policies = [-2] * player_count
            policies[learner] = -1
            return policies
        if self.rng.random() < self.args.shared_selfplay_fraction:
            return [-1] * player_count
        policies = [int(self.rng.integers(0, len(self.snapshots))) for _ in range(player_count)]
        policies[learner] = -1
        return policies

    def _new_env(self, game_index: int) -> tuple[MonopolyEnv, list[int], np.ndarray]:
        progress = min(1.0, self.state.games / max(1, self.args.curriculum_games))
        curriculum = round(self.args.initial_rounds + progress * (self.args.max_rounds - self.args.initial_rounds))
        env = MonopolyEnv(self._player_count(), self.args.seed + self.state.games + game_index * 7919, max_rounds=curriculum)
        policies = self._seat_policies(env.player_count, self.state.games + game_index)
        hidden = np.zeros((MAX_PLAYERS, self.model.hidden_size), dtype=np.float32)
        return env, policies, hidden

    def collect_games(self) -> tuple[list[Transition], dict[str, float]]:
        target = self.args.games_per_update
        active_count = min(self.args.envs, target)
        envs: list[MonopolyEnv] = []
        policies: list[list[int]] = []
        hidden_states: list[np.ndarray] = []
        trajectories: list[list[list[Transition]]] = []
        for index in range(active_count):
            env, seat_policies, hidden = self._new_env(index)
            envs.append(env)
            policies.append(seat_policies)
            hidden_states.append(hidden)
            trajectories.append([[] for _ in range(MAX_PLAYERS)])

        completed = 0
        launched = active_count
        output: list[Transition] = []
        learner_outcomes: list[float] = []
        wins = 0
        league_wins = 0
        league_total = 0
        self.model.eval()
        while completed < target:
            rows_by_policy: dict[int, list[int]] = {}
            observations: list[dict[str, np.ndarray]] = []
            actions_by_env: list[list[Action]] = []
            actors: list[int] = []
            for row, env in enumerate(envs):
                observation = env.observe()
                actions = env.legal_actions()
                actor = env.actor
                observations.append(observation)
                actions_by_env.append(actions)
                actors.append(actor)
                policy_id = policies[row][actor]
                rows_by_policy.setdefault(policy_id, []).append(row)

            selected = [0] * len(envs)
            log_probs = [0.0] * len(envs)
            values = [0.0] * len(envs)
            hidden_before: list[np.ndarray | None] = [None] * len(envs)
            for policy_id, rows in rows_by_policy.items():
                if policy_id == -2:
                    for row in rows:
                        selected[row] = int(envs[row].rng.integers(len(actions_by_env[row])))
                    continue
                policy = self.model if policy_id == -1 else self.snapshots[policy_id]
                obs_batch = collate_observations([observations[row] for row in rows], self.device)
                action_batch, mask = collate_actions(
                    [actions_by_env[row] for row in rows],
                    [actors[row] for row in rows],
                    [envs[row].player_count for row in rows],
                    self.device,
                )
                hidden_batch = torch.as_tensor(
                    np.stack([hidden_states[row][actors[row]] for row in rows]), device=self.device
                )
                with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                    logits, value, next_hidden = policy(obs_batch, action_batch, mask, hidden_batch)
                    distribution = Categorical(logits=logits.float())
                    choice = distribution.sample()
                    log_prob = distribution.log_prob(choice)
                for offset, row in enumerate(rows):
                    selected[row] = int(choice[offset].item())
                    log_probs[row] = float(log_prob[offset].item())
                    values[row] = float(value[offset].item())
                    hidden_before[row] = hidden_states[row][actors[row]].copy()
                    hidden_states[row][actors[row]] = next_hidden[offset].float().cpu().numpy()

            finished_rows: list[int] = []
            for row, env in enumerate(envs):
                actor = actors[row]
                policy_id = policies[row][actor]
                index = selected[row]
                if policy_id == -1:
                    action_features = env.action_features(actions_by_env[row], actor, env.player_count)
                    trajectories[row][actor].append(
                        Transition(
                            observation={key: value.copy() for key, value in observations[row].items()},
                            action_features={key: value.copy() for key, value in action_features.items()},
                            action_index=index,
                            old_log_prob=log_probs[row],
                            old_value=values[row],
                            hidden=np.asarray(hidden_before[row], dtype=np.float16),
                        )
                    )
                _, _, done, info = env.step(actions_by_env[row][index], validate=False)
                self.state.decisions += 1
                if not done:
                    continue
                rewards = np.asarray(info["terminal_rewards"], dtype=np.float32)
                learner_players = [player for player, policy in enumerate(policies[row]) if policy == -1]
                if len(learner_players) == 1 and any(policy >= 0 for policy in policies[row]):
                    league_total += 1
                    league_wins += int(rewards[learner_players[0]] >= 0.999)
                for player in learner_players:
                    outcome = float(rewards[player])
                    learner_outcomes.append(outcome)
                    wins += int(outcome >= 0.999)
                    for transition in trajectories[row][player]:
                        transition.outcome = outcome
                        output.append(transition)
                completed += 1
                finished_rows.append(row)

            for row in reversed(finished_rows):
                if launched < target:
                    env, seat_policies, hidden = self._new_env(launched)
                    envs[row] = env
                    policies[row] = seat_policies
                    hidden_states[row] = hidden
                    trajectories[row] = [[] for _ in range(MAX_PLAYERS)]
                    launched += 1
                else:
                    del envs[row]
                    del policies[row]
                    del hidden_states[row]
                    del trajectories[row]

        self.state.games += completed
        if len(output) > self.args.max_transitions:
            indices = self.rng.choice(len(output), self.args.max_transitions, replace=False)
            output = [output[int(index)] for index in indices]
        total = len(learner_outcomes)
        return output, {
            "games": float(completed),
            "transitions": float(len(output)),
            "mean_outcome": float(np.mean(learner_outcomes)) if learner_outcomes else 0.0,
            "win_rate": wins / max(1, total),
            "win_rate_lower": wilson_lower_bound(wins, total),
            "league_wins": float(league_wins),
            "league_total": float(league_total),
            "league_win_rate": league_wins / max(1, league_total),
            "league_win_rate_lower": wilson_lower_bound(league_wins, league_total),
        }

    def _collate_transitions(self, batch: list[Transition]) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor]:
        observations = collate_observations([item.observation for item in batch], self.device)
        actions, mask = collate_action_feature_rows(
            [item.action_features for item in batch], self.device
        )
        return (
            observations,
            actions,
            mask,
            torch.as_tensor([item.action_index for item in batch], device=self.device),
            torch.as_tensor([item.old_log_prob for item in batch], dtype=torch.float32, device=self.device),
            torch.as_tensor([item.outcome for item in batch], dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack([item.hidden for item in batch]), dtype=torch.float32, device=self.device),
        )

    def ppo_update(self, transitions: list[Transition]) -> dict[str, float]:
        self.model.train()
        old_values = np.asarray([item.old_value for item in transitions], dtype=np.float32)
        outcomes = np.asarray([item.outcome for item in transitions], dtype=np.float32)
        advantages = outcomes - old_values
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
        for item, advantage in zip(transitions, advantages, strict=True):
            item.old_value = float(advantage)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "batches": 0}
        for _epoch in range(self.args.ppo_epochs):
            order = self.rng.permutation(len(transitions))
            for start in range(0, len(order), self.args.minibatch):
                indices = order[start : start + self.args.minibatch]
                batch = [transitions[int(index)] for index in indices]
                obs, actions, mask, chosen, old_log_prob, returns, hidden = self._collate_transitions(batch)
                advantage = torch.as_tensor([item.old_value for item in batch], dtype=torch.float32, device=self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                    logits, values, _ = self.model(obs, actions, mask, hidden)
                    distribution = Categorical(logits=logits.float())
                    log_prob = distribution.log_prob(chosen)
                    entropy = distribution.entropy().mean()
                    ratio = (log_prob - old_log_prob).exp()
                    unclipped = ratio * advantage
                    clipped = ratio.clamp(1 - self.args.clip, 1 + self.args.clip) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (values.float() - returns).square().mean()
                    loss = policy_loss + self.args.value_coef * value_loss - self.args.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                approximate_kl = float((old_log_prob - log_prob.detach()).mean().item())
                totals["policy_loss"] += float(policy_loss.detach().item())
                totals["value_loss"] += float(value_loss.detach().item())
                totals["entropy"] += float(entropy.detach().item())
                totals["kl"] += approximate_kl
                totals["batches"] += 1
            if totals["batches"] and totals["kl"] / totals["batches"] > self.args.target_kl:
                break
        batches = max(1, totals.pop("batches"))
        return {key: value / batches for key, value in totals.items()}

    def run(self) -> None:
        if self.pause_path.exists():
            self.pause_path.unlink()
        self._write_status("starting")
        high_temperature_checks = 0
        last_checkpoint = time.monotonic()
        while self.args.max_updates <= 0 or self.state.update < self.args.max_updates:
            started = time.monotonic()
            transitions, collection = self.collect_games()
            if not transitions:
                raise RuntimeError("rollout produced no learner transitions")
            learning = self.ppo_update(transitions)
            self.state.update += 1
            self.state.league_window_wins += int(collection["league_wins"])
            self.state.league_window_total += int(collection["league_total"])
            if self.state.league_window_total > 2_000:
                self.state.league_window_wins //= 2
                self.state.league_window_total //= 2
            score = self.state.league_window_wins / max(1, self.state.league_window_total)
            league_lower = wilson_lower_bound(self.state.league_window_wins, self.state.league_window_total)
            if score > self.state.best_score + self.args.min_improvement:
                self.state.best_score = score
                self.state.no_improvement = 0
            else:
                self.state.no_improvement += 1
            self.scheduler.step(score)
            if self.state.update % self.args.snapshot_every == 0:
                self._add_snapshot()

            temperature = gpu_temperature()
            high_temperature_checks = high_temperature_checks + 1 if temperature is not None and temperature >= self.args.max_gpu_temp else 0
            metrics = {
                "update": self.state.update,
                "total_games": self.state.games,
                "total_decisions": self.state.decisions,
                "seconds": time.monotonic() - started,
                "gpu_temperature": temperature,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "league_window_win_rate": score,
                "league_window_win_rate_lower": league_lower,
                **collection,
                **learning,
            }
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            self._write_status("running", last_metrics=metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)

            checkpoint_due = time.monotonic() - last_checkpoint >= self.args.checkpoint_minutes * 60
            pause_requested = STOP_REQUESTED or self.pause_path.exists()
            thermal_stop = high_temperature_checks >= 3
            adaptive_stop = (
                self.state.games >= self.args.min_games
                and league_lower >= self.args.league_win_rate
                and self.state.no_improvement >= self.args.patience
                and self.optimizer.param_groups[0]["lr"] <= 1.01e-5
            )
            if checkpoint_due or pause_requested or thermal_stop or adaptive_stop:
                reason = "pause" if pause_requested else "thermal" if thermal_stop else "adaptive_stop" if adaptive_stop else "periodic"
                self._save_checkpoint(reason)
                last_checkpoint = time.monotonic()
            if pause_requested or thermal_stop or adaptive_stop:
                state = "paused" if pause_requested else "thermal_paused" if thermal_stop else "complete"
                self._write_status(state, reason=reason, last_metrics=metrics)
                if self.pause_path.exists():
                    self.pause_path.unlink()
                return
        self._save_checkpoint("maximum_updates")
        self._write_status("complete", reason="maximum_updates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-knowledge league self-play trainer for MonopolyAI")
    parser.add_argument("--run-dir", default="training/runs/zero-knowledge-main")
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--games-per-update", type=int, default=16)
    parser.add_argument("--max-transitions", type=int, default=65536)
    parser.add_argument("--initial-rounds", type=int, default=24)
    parser.add_argument("--rounds-growth", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--curriculum-games", type=int, default=250_000)
    parser.add_argument("--max-rounds", type=int, default=360)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--max-grad-norm", type=float, default=0.7)
    parser.add_argument("--target-kl", type=float, default=0.025)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--max-snapshot-files", type=int, default=48)
    parser.add_argument("--random-opponent-fraction", type=float, default=0.25)
    parser.add_argument("--shared-selfplay-fraction", type=float, default=0.2)
    parser.add_argument("--checkpoint-minutes", type=float, default=20)
    parser.add_argument("--max-gpu-temp", type=int, default=80)
    parser.add_argument("--min-games", type=int, default=2_000_000)
    parser.add_argument("--league-win-rate", type=float, default=0.55)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-from", default="", help="initialize a player-count fine-tune from another checkpoint")
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    trainer = LeagueTrainer(parse_args())
    trainer.run()


if __name__ == "__main__":
    main()
