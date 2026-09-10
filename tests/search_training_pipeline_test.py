from __future__ import annotations

import unittest
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from training.search_pipeline import (
    EvaluationSchedule,
    LeagueRoster,
    ConfiguredSearchPolicy,
    build_balanced_evaluation_schedule,
    evaluate_policy_league,
    freeze_due_milestones,
    load_search_config,
    load_frozen_policy,
    run_league_training,
    teacher_generation_defaults,
    SearchTeacher,
    SearchRunControl,
    SearchRunState,
    SeatRuntime,
    generate_teacher_episode,
    read_teacher_dataset,
    sample_public_belief,
    public_view,
    write_teacher_dataset,
)
from training.search_pipeline.distill import distill_batch
from training.search_pipeline.evaluation import checkpoint_artifact, parse_named_checkpoints
from training.search_pipeline.league import _update_candidate
from training.search_pipeline.policies import LegacyNeuroPolicy
from training.zero_knowledge.model import ModelConfig
from training.zero_knowledge.model import PropertyTradingPolicy
from training.zero_knowledge.env import Action, ActionKind, PropertyTradingEnv


class SearchTrainingPipelineTests(unittest.TestCase):
    def test_public_view_contains_only_observation_and_legal_candidates(self) -> None:
        env = PropertyTradingEnv(4, seed=91, max_rounds=4)
        view = public_view(env, seat=env.actor)
        self.assertEqual(
            set(view),
            {"actor", "seat", "player_count", "observation", "legal_actions"},
        )
        self.assertEqual(view["actor"], env.actor)
        self.assertEqual(view["legal_actions"], env.legal_actions())
        self.assertNotIn("rng", view)
        self.assertNotIn("chance_deck", view)

    def test_each_seat_owns_independent_reproducible_memory_reward_and_rng(self) -> None:
        first = [SeatRuntime.create(101, seat) for seat in range(4)]
        second = [SeatRuntime.create(101, seat) for seat in range(4)]
        self.assertEqual(
            [runtime.rng.integers(1_000_000) for runtime in first],
            [runtime.rng.integers(1_000_000) for runtime in second],
        )
        self.assertEqual(len({runtime.seed for runtime in first}), 4)
        first[0].memory["turns"] = 1
        first[0].reward = 0.5
        self.assertEqual(first[1].memory, {})
        self.assertEqual(first[1].reward, 0.0)

    def test_search_teacher_is_seeded_and_can_only_return_a_legal_action(self) -> None:
        left = PropertyTradingEnv(3, seed=17, max_rounds=3)
        right = PropertyTradingEnv(3, seed=17, max_rounds=3)
        teacher_a = SearchTeacher(seed=2026, simulations=6, depth=8)
        teacher_b = SearchTeacher(seed=2026, simulations=6, depth=8)
        action_a, policy_a = teacher_a.select_action(left)
        action_b, policy_b = teacher_b.select_action(right)
        self.assertIn(action_a, left.legal_actions())
        self.assertEqual(action_a, action_b)
        np.testing.assert_allclose(policy_a, policy_b)
        self.assertAlmostEqual(float(policy_a.sum()), 1.0)

    def test_rollout_belief_does_not_depend_on_hidden_future_card_order(self) -> None:
        left = PropertyTradingEnv(4, seed=91, max_rounds=4)
        right = PropertyTradingEnv(4, seed=91, max_rounds=4)
        right.chance_deck.reverse()
        right.community_deck.reverse()

        sampled_left = sample_public_belief(left, seed=20260910)
        sampled_right = sample_public_belief(right, seed=20260910)

        self.assertEqual(sampled_left.chance_deck, sampled_right.chance_deck)
        self.assertEqual(sampled_left.community_deck, sampled_right.community_deck)
        self.assertNotEqual(sampled_left.chance_deck, left.chance_deck)

    def test_gpu_hour_milestones_fire_once_at_100_300_and_500_hours(self) -> None:
        schedule = EvaluationSchedule()
        self.assertEqual(schedule.due(99.9), [])
        self.assertEqual(schedule.due(100), [100])
        self.assertEqual(schedule.due(349), [300])
        self.assertEqual(schedule.due(600), [500])
        self.assertEqual(schedule.due(900), [])

    def test_evaluation_balances_candidate_seats_and_reports_release_metrics(self) -> None:
        schedule = build_balanced_evaluation_schedule(
            player_counts=(3, 4, 5),
            seeds=(11,),
            candidate="student-v1",
            opponents=("baseline-a", "baseline-b"),
        )
        for player_count in (3, 4, 5):
            seats = [
                game.candidate_seat
                for game in schedule
                if game.player_count == player_count
            ]
            self.assertEqual(
                seats,
                [seat for seat in range(player_count) for _ in range(2)],
            )

            for physical_seat in range(player_count):
                counts = {"baseline-a": 0, "baseline-b": 0}
                for game in schedule:
                    if game.player_count != player_count:
                        continue
                    policy_name = game.seat_policies[physical_seat]
                    if policy_name in counts:
                        counts[policy_name] += 1
                self.assertEqual(counts["baseline-a"], counts["baseline-b"])

        def first_legal(view: dict[str, object]):
            return view["legal_actions"][0]

        report = evaluate_policy_league(
            schedule,
            {
                "student-v1": first_legal,
                "baseline-a": first_legal,
                "baseline-b": first_legal,
            },
            max_rounds=1,
        )
        self.assertEqual(report["games"], 24)
        self.assertEqual(report["illegalActions"], 0)
        self.assertIn("winRate", report)
        self.assertIn("averageRank", report)
        self.assertEqual(len(report["wilson95"]), 2)
        self.assertGreaterEqual(report["decisionTimeMs"]["p95"], 0.0)

        opponent_counts = {"baseline-a": 0, "baseline-b": 0}
        for game in schedule:
            for policy_name in game.seat_policies:
                if policy_name in opponent_counts:
                    opponent_counts[policy_name] += 1
        self.assertEqual(opponent_counts["baseline-a"], opponent_counts["baseline-b"])

    def test_checkpoint_artifact_uses_saved_policy_identity_and_content_hash(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            torch.save({"policy_version": "candidate-v7", "model_config": {}}, checkpoint)
            artifact = checkpoint_artifact(checkpoint, name="candidate")
            self.assertEqual(artifact["name"], "candidate")
            self.assertEqual(artifact["policyVersion"], "candidate-v7")
            self.assertEqual(artifact["path"], str(checkpoint.resolve()))
            self.assertEqual(len(artifact["sha256"]), 64)

        legacy = checkpoint_artifact(Path("app/pretrained-model.json"), name="legacy")
        self.assertEqual(
            [file["playerCount"] for file in legacy["files"]],
            [3, 4, 5],
        )
        self.assertTrue(all(len(file["sha256"]) == 64 for file in legacy["files"]))

    def test_evaluation_rejects_duplicate_or_candidate_colliding_opponent_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate opponent"):
            parse_named_checkpoints(
                ["baseline=a.pt", "baseline=b.pt"],
                candidate_name="candidate-v1",
            )
        with self.assertRaisesRegex(ValueError, "conflicts with candidate"):
            parse_named_checkpoints(
                ["candidate-v1=a.pt"],
                candidate_name="candidate-v1",
            )

    def test_due_milestones_freeze_checkpoint_report_and_run_state(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "student.pt"
            checkpoint.write_bytes(b"candidate-v1")
            state = SearchRunState(gpu_seconds=101 * 3600)
            report = {"games": 12, "winRate": 0.5, "illegalActions": 0}

            frozen = freeze_due_milestones(
                state=state,
                checkpoint=checkpoint,
                run_directory=root / "run",
                report=report,
            )

            self.assertEqual([item["milestoneGpuHours"] for item in frozen], [100])
            self.assertEqual(state.completed_milestones, [100])
            frozen_checkpoint = root / "run" / "milestones" / "100-gpu-hours" / "candidate.pt"
            self.assertEqual(frozen_checkpoint.read_bytes(), b"candidate-v1")
            saved_report = json.loads(frozen_checkpoint.with_name("evaluation.json").read_text("utf-8"))
            self.assertEqual(saved_report["milestoneGpuHours"], 100)

            skipped_state = SearchRunState(gpu_seconds=501 * 3600)
            with self.assertRaisesRegex(RuntimeError, "missed evaluation milestones"):
                freeze_due_milestones(
                    state=skipped_state,
                    checkpoint=checkpoint,
                    run_directory=root / "skipped",
                    report=report,
                )
            self.assertFalse((root / "skipped" / "milestones").exists())

    def test_teacher_dataset_and_frozen_league_roster_are_reproducible(self) -> None:
        samples_a = generate_teacher_episode(
            player_count=3,
            game_seed=71,
            teacher_seed=72,
            simulations=2,
            depth=2,
            max_rounds=2,
        )
        samples_b = generate_teacher_episode(
            player_count=3,
            game_seed=71,
            teacher_seed=72,
            simulations=2,
            depth=2,
            max_rounds=2,
        )
        self.assertEqual(samples_a, samples_b)
        self.assertTrue(samples_a)
        self.assertNotIn("rng", samples_a[0])
        self.assertEqual(samples_a[0]["game_seed"], 71)
        self.assertEqual(samples_a[0]["teacher_seed"], 72)
        self.assertEqual(samples_a[0]["teacher_version"], "seeded-stochastic-rollout-v1")
        self.assertEqual(samples_a[0]["teacher_config"], {
            "simulations": 2,
            "depth": 2,
            "max_rounds": 2,
        })
        self.assertEqual(
            [sample["decision_index"] for sample in samples_a],
            list(range(len(samples_a))),
        )
        self.assertEqual(
            [sample["selected_action"] for sample in samples_a],
            [
                sample["legal_actions"].index(sample["action"])
                for sample in samples_a
            ],
        )
        self.assertAlmostEqual(sum(samples_a[0]["policy"]), 1.0, places=5)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            write_teacher_dataset(path, samples_a)
            self.assertEqual(read_teacher_dataset(path), samples_a)

        roster = LeagueRoster(seed=73)
        roster.add("neuroevolution-v5", "checkpoints/neuro-v5.pt", frozen=True)
        roster.add("zero-knowledge-ppo-v1", "checkpoints/ppo-v1.pt", frozen=True)
        self.assertEqual(roster.sample(2), ["neuroevolution-v5", "zero-knowledge-ppo-v1"])
        with self.assertRaises(ValueError):
            roster.add("neuroevolution-v5", "other.pt", frozen=True)
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "league.json"
            roster.save(manifest)
            restored = LeagueRoster.load(manifest, seed=73)
            self.assertEqual(restored.entries, roster.entries)

        metrics = distill_batch(
            samples_a[:8],
            config=ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96),
            steps=1,
            device="cpu",
            seed=74,
        )
        self.assertGreaterEqual(metrics["policy_loss"], 0.0)
        self.assertGreaterEqual(metrics["value_loss"], 0.0)
        repeated_metrics = distill_batch(
            samples_a[:8],
            config=ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96),
            steps=1,
            device="cpu",
            seed=74,
        )
        self.assertEqual(metrics, repeated_metrics)

    def test_search_run_state_can_pause_and_resume_without_losing_milestones(self) -> None:
        with TemporaryDirectory() as directory:
            control = SearchRunControl(Path(directory))
            state = SearchRunState(completed_games=12, gpu_seconds=360_000, completed_milestones=[100])
            control.save(state)
            control.request_pause()
            self.assertTrue(control.pause_requested())
            self.assertEqual(control.load(), state)
            control.clear_pause()
            self.assertFalse(control.pause_requested())

    def test_beginning_a_paused_search_run_consumes_the_stale_pause_request(self) -> None:
        with TemporaryDirectory() as directory:
            control = SearchRunControl(Path(directory))
            state = SearchRunState(completed_games=7)
            control.save(state)
            control.request_pause()

            self.assertEqual(control.begin(), state)
            self.assertFalse(control.pause_requested())

    def test_league_update_replays_the_hidden_context_used_during_sampling(self) -> None:
        config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
        model = PropertyTradingPolicy(config)
        env = PropertyTradingEnv(3, seed=80, max_rounds=1)
        view = public_view(env, env.actor)
        expected_hidden = torch.full((config.d_model,), 0.25)
        captured: list[torch.Tensor | None] = []
        original_forward = model.forward

        def recording_forward(observation, actions, action_mask, hidden=None):
            captured.append(None if hidden is None else hidden.detach().clone())
            return original_forward(observation, actions, action_mask, hidden)

        model.forward = recording_forward
        _update_candidate(model, [{
            "observation": view["observation"],
            "legal_actions": view["legal_actions"],
            "actor": view["actor"],
            "player_count": view["player_count"],
            "selected_action": 0,
            "outcome": 1.0,
            "hidden": expected_hidden,
        }], torch.device("cpu"))

        self.assertIsNotNone(captured[0])
        torch.testing.assert_close(captured[0], expected_hidden.unsqueeze(0))

    def test_league_self_play_uses_frozen_checkpoint_opponents_and_isolated_seats(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(81)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save({
                "version": 1,
                "stage": "search-teacher-distillation",
                "model_config": asdict(config),
                "model": model.state_dict(),
            }, base)
            roster = LeagueRoster(seed=82)
            roster.add("frozen-v1", str(base), frozen=True)
            output = root / "league.pt"

            metrics = run_league_training(
                checkpoint=base,
                roster=roster,
                output=output,
                games=3,
                player_counts=(3,),
                seed=83,
                max_rounds=1,
                device="cpu",
            )

            self.assertTrue(output.is_file())
            self.assertEqual(metrics["games"], 3)
            self.assertEqual(metrics["candidateSeats"], [0, 1, 2])
            self.assertEqual(len(set(metrics["seatSeeds"][0])), 3)
            self.assertEqual(sorted(metrics["seatRewards"][0]), [-1.0, 0.0, 1.0])
            payload = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(payload["stage"], "league-self-play")
            self.assertEqual(payload["policy_version"], "search-student-v1")
            self.assertEqual(payload["league"][0]["name"], "frozen-v1")

    def test_league_rotates_multiple_opponents_across_physical_seats(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(92)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save({
                "version": 1,
                "model_config": asdict(config),
                "model": model.state_dict(),
            }, base)
            roster = LeagueRoster(seed=93)
            roster.add("frozen-a", str(base), frozen=True)
            roster.add("frozen-b", str(base), frozen=True)

            metrics = run_league_training(
                checkpoint=base,
                roster=roster,
                output=root / "league.pt",
                games=6,
                player_counts=(3,),
                seed=94,
                max_rounds=1,
                device="cpu",
            )

            self.assertEqual(metrics["candidateSeats"], [0, 1, 2, 0, 1, 2])
            for physical_seat in range(3):
                names = [
                    seats[physical_seat]
                    for seats in metrics["seatPolicies"]
                    if seats[physical_seat] != "candidate"
                ]
                self.assertEqual(names.count("frozen-a"), names.count("frozen-b"))

    def test_legacy_policy_requires_matching_player_count_and_uses_action_thresholds(self) -> None:
        legacy = LegacyNeuroPolicy(Path("app/pretrained-model-3p.json"))
        env = PropertyTradingEnv(3, seed=90, max_rounds=1)
        self.assertIn(legacy(public_view(env, env.actor)), env.legal_actions())

        wrong_count = PropertyTradingEnv(4, seed=90, max_rounds=1)
        with self.assertRaisesRegex(ValueError, "4-player"):
            legacy(public_view(wrong_count, wrong_count.actor))

        payload = json.loads(Path("app/pretrained-model-3p.json").read_text("utf-8"))
        payload["outputBias"] = [10.0, -10.0, -10.0, -10.0, 10.0, -10.0]
        payload["inputHidden"] = [0.0] * len(payload["inputHidden"])
        payload["hiddenOutput"] = [0.0] * len(payload["hiddenOutput"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reserve-heavy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            policy = LegacyNeuroPolicy(path)
            env = PropertyTradingEnv(3, seed=91, max_rounds=1)
            env.cash[env.actor] = 60
            env.pending_tile = 1
            env.phase = "buy"
            action = policy(public_view(env, env.actor))
            self.assertEqual(action.kind.name, "DECLINE")

        trade_env = PropertyTradingEnv(3, seed=95, max_rounds=2)
        trade_env.owner[1] = 0
        trade_env.owner[39] = 1
        trade_env.phase = "turn_start"
        trade_env.actor = 0
        trade_env.active = 0
        trade_view = public_view(trade_env, 0)
        trade = next(
            action
            for action in trade_view["legal_actions"]
            if action.kind.name == "TRADE" and action.take_tile == 39 and action.cash > 0
        )
        features = legacy._features(trade_view, trade)
        self.assertGreater(features[2], 0.0)
        self.assertGreater(features[3], 0.0)

        response_env = PropertyTradingEnv(3, seed=96, max_rounds=2)
        response_env.owner[1] = 1
        response_env.owner[3] = 1
        response_env.active = 0
        response_env.actor = 1
        response_env.phase = "trade_response"
        response_env.pending_trade = Action(
            ActionKind.TRADE,
            target=1,
            cash=20,
            take_tile=3,
        )
        response_features = legacy._trade_response_features(public_view(response_env, 1))
        self.assertAlmostEqual(response_features[4], 0.5)
        self.assertLess(response_features[3], 0.0)

    def test_versioned_training_config_drives_fast_and_upper_bound_policies(self) -> None:
        config = load_search_config()
        self.assertEqual(config["student"]["version"], "search-student-v1")
        self.assertEqual(config["policies"]["fastPlay"]["version"], "search-student-v1")
        self.assertGreater(config["policies"]["upperBound"]["simulations"], 0)
        self.assertEqual(
            teacher_generation_defaults(),
            {
                "simulations": config["teacher"]["simulations"],
                "depth": config["teacher"]["depth"],
            },
        )

        env = PropertyTradingEnv(3, seed=87, max_rounds=1)
        legacy = load_frozen_policy(
            Path("app/pretrained-model-3p.json"),
            device="cpu",
        )
        self.assertIn(legacy(public_view(env, env.actor)), env.legal_actions())

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            small = json.loads(json.dumps(config))
            small["policies"]["upperBound"]["simulations"] = 1
            small["policies"]["upperBound"]["depth"] = 1
            config_path.write_text(json.dumps(small), encoding="utf-8")
            upper = ConfiguredSearchPolicy(seed=88, config_path=config_path)
            self.assertEqual(
                upper.provenance(),
                {
                    "policyVersion": small["policies"]["upperBound"]["version"],
                    "teacherSeed": 88,
                    "simulations": 1,
                    "depth": 1,
                    "configVersion": small["version"],
                    "configPath": str(config_path.resolve()),
                },
            )
            schedule = build_balanced_evaluation_schedule(
                player_counts=(3,),
                seeds=(89,),
                candidate="upper",
                opponents=("baseline",),
            )
            report = evaluate_policy_league(
                schedule,
                {"upper": upper, "baseline": lambda view: view["legal_actions"][0]},
                max_rounds=1,
            )
            self.assertEqual(report["games"], 3)


if __name__ == "__main__":
    unittest.main()
