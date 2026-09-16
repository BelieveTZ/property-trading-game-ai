from __future__ import annotations

import unittest
import json
import subprocess
import sys
import time
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
    TeacherEpisodePaused,
    generate_teacher_episode,
    generate_teacher_episode_file,
    generate_teacher_games,
    read_teacher_dataset,
    sample_public_belief,
    public_view,
    write_teacher_dataset,
)
from training.search_pipeline.distill import distill_batch, distillation_defaults
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

    def test_search_teacher_limits_candidates_to_configured_budget(self) -> None:
        env = PropertyTradingEnv(3, seed=18, max_rounds=3)
        env.owner[1] = 0
        env.owner[39] = 1
        env.active = 0
        env.actor = 0
        env.phase = "turn_start"
        legal_actions = env.legal_actions()
        self.assertGreater(len(legal_actions), 4)

        action, policy = SearchTeacher(
            seed=2027,
            simulations=4,
            depth=1,
        ).select_action(env)

        self.assertIn(action, legal_actions)
        self.assertEqual(policy.shape, (len(legal_actions),))
        self.assertLessEqual(int(np.count_nonzero(policy)), 4)
        self.assertAlmostEqual(float(policy.sum()), 1.0)

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
        self.assertEqual(samples_a[0]["teacher_version"], "seeded-candidate-rollout-v2")
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

    def test_teacher_dataset_stores_only_evaluated_action_candidates(self) -> None:
        samples = generate_teacher_episode(
            player_count=3,
            game_seed=75,
            teacher_seed=76,
            simulations=2,
            depth=1,
            max_rounds=1,
        )

        self.assertTrue(samples)
        self.assertTrue(
            all(len(sample["legal_actions"]) <= 2 for sample in samples),
        )
        self.assertTrue(
            all(len(sample["policy"]) == len(sample["legal_actions"]) for sample in samples),
        )
        self.assertTrue(
            all(
                sample["action"] == sample["legal_actions"][sample["selected_action"]]
                for sample in samples
            ),
        )

    def test_distill_cli_trains_a_dataset_larger_than_its_batch(self) -> None:
        samples = generate_teacher_episode(
            player_count=3,
            game_seed=177,
            teacher_seed=178,
            simulations=2,
            depth=1,
            max_rounds=1,
        )[:6]
        self.assertEqual(len(samples), 6)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.jsonl"
            output = root / "student.pt"
            run_directory = root / "run"
            write_teacher_dataset(dataset, samples)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.search_pipeline.distill",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--run-dir",
                    str(run_directory),
                    "--steps",
                    "2",
                    "--batch-size",
                    "2",
                    "--checkpoint-every",
                    "1",
                    "--device",
                    "cpu",
                    "--d-model",
                    "24",
                    "--heads",
                    "4",
                    "--layers",
                    "1",
                    "--feedforward",
                    "48",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = torch.load(output, map_location="cpu", weights_only=False)
            state = json.loads(
                (run_directory / "distill-state.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(payload["samples"], 6)
            self.assertEqual(payload["training_batch_size"], 2)
            self.assertEqual(payload["training_steps"], 2)
            self.assertEqual(payload["sampling_version"], "seeded-with-replacement-v1")
            self.assertEqual(payload["optimizer"], {
                "name": "AdamW",
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
            })
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["completed_steps"], 2)
            self.assertEqual(state["total_steps"], 2)
            self.assertTrue((run_directory / "distill-checkpoint.pt").is_file())
            manifest = json.loads(
                (run_directory / "training-command.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(manifest["version"], 1)
            self.assertTrue(Path(manifest["executable"]).samefile(sys.executable))
            self.assertEqual(manifest["arguments"][:2], ["-m", "training.search_pipeline.distill"])
            self.assertEqual(manifest["pidFile"], "distill-pid.txt")
            self.assertEqual(manifest["pauseFile"], "distill-pause.request")
            self.assertEqual(manifest["stateFile"], "distill-state.json")

    def test_distill_cli_checkpoints_and_exits_when_pause_is_requested(self) -> None:
        samples = generate_teacher_episode(
            player_count=3,
            game_seed=182,
            teacher_seed=183,
            simulations=2,
            depth=1,
            max_rounds=1,
        )[:6]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.jsonl"
            output = root / "student.pt"
            run_directory = root / "run"
            write_teacher_dataset(dataset, samples)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "training.search_pipeline.distill",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--run-dir",
                    str(run_directory),
                    "--steps",
                    "1000",
                    "--batch-size",
                    "2",
                    "--checkpoint-every",
                    "1000",
                    "--device",
                    "cpu",
                    "--d-model",
                    "24",
                    "--heads",
                    "4",
                    "--layers",
                    "1",
                    "--feedforward",
                    "48",
                ],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                state_path = run_directory / "distill-state.json"
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not state_path.exists():
                    self.assertIsNone(process.poll(), "distillation exited before starting")
                    time.sleep(0.02)
                self.assertTrue(state_path.exists())
                (run_directory / "distill-pause.request").touch()
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr or stdout)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "paused")
            self.assertGreater(state["completed_steps"], 0)
            self.assertLess(state["completed_steps"], 1000)
            self.assertTrue((run_directory / "distill-checkpoint.pt").is_file())
            self.assertFalse(output.exists())

    def test_distill_cli_resumes_deterministically_from_its_checkpoint(self) -> None:
        samples = generate_teacher_episode(
            player_count=3,
            game_seed=179,
            teacher_seed=180,
            simulations=2,
            depth=1,
            max_rounds=1,
        )[:6]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.jsonl"
            resumed_output = root / "resumed.pt"
            uninterrupted_output = root / "uninterrupted.pt"
            resumed_run = root / "resumed-run"
            uninterrupted_run = root / "uninterrupted-run"
            write_teacher_dataset(dataset, samples)

            def command(output: Path, run_directory: Path, steps: int) -> list[str]:
                return [
                    sys.executable,
                    "-m",
                    "training.search_pipeline.distill",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--run-dir",
                    str(run_directory),
                    "--steps",
                    str(steps),
                    "--batch-size",
                    "2",
                    "--checkpoint-every",
                    "1",
                    "--device",
                    "cpu",
                    "--d-model",
                    "24",
                    "--heads",
                    "4",
                    "--layers",
                    "1",
                    "--feedforward",
                    "48",
                    "--seed",
                    "181",
                ]

            for invocation in (
                command(resumed_output, resumed_run, 1),
                command(resumed_output, resumed_run, 2),
                command(uninterrupted_output, uninterrupted_run, 2),
            ):
                result = subprocess.run(
                    invocation,
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            state = json.loads(
                (resumed_run / "distill-state.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(state["resumed_from_step"], 1)
            resumed = torch.load(resumed_output, map_location="cpu", weights_only=False)
            uninterrupted = torch.load(
                uninterrupted_output,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(resumed["metrics"], uninterrupted["metrics"])
            for name, expected in uninterrupted["model"].items():
                self.assertTrue(torch.equal(resumed["model"][name], expected), name)

    def test_teacher_episode_file_resumes_from_an_intra_game_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "game-00000.jsonl"
            checkpoint = root / "game-00000.checkpoint"

            with self.assertRaises(TeacherEpisodePaused):
                generate_teacher_episode_file(
                    output=output,
                    checkpoint=checkpoint,
                    player_count=3,
                    game_seed=77,
                    teacher_seed=78,
                    simulations=1,
                    depth=1,
                    max_rounds=1,
                    checkpoint_every=2,
                    should_pause=lambda decisions: decisions >= 2,
                )

            self.assertTrue(checkpoint.is_file())
            self.assertTrue(output.with_suffix(".partial.jsonl").is_file())

            decision_count = generate_teacher_episode_file(
                output=output,
                checkpoint=checkpoint,
                player_count=3,
                game_seed=77,
                teacher_seed=78,
                simulations=1,
                depth=1,
                max_rounds=1,
                checkpoint_every=2,
            )

            self.assertEqual(decision_count, len(read_teacher_dataset(output)))
            self.assertEqual(
                read_teacher_dataset(output),
                generate_teacher_episode(
                    player_count=3,
                    game_seed=77,
                    teacher_seed=78,
                    simulations=1,
                    depth=1,
                    max_rounds=1,
                ),
            )
            self.assertFalse(checkpoint.exists())
            self.assertFalse(output.with_suffix(".partial.jsonl").exists())

    def test_teacher_games_generate_parallel_shards_and_merge_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "teacher.jsonl"
            run_directory = root / "run"

            state = generate_teacher_games(
                output=output,
                run_directory=run_directory,
                player_count=3,
                games=2,
                game_seed=79,
                teacher_seed=80,
                simulations=1,
                depth=1,
                max_rounds=1,
                workers=2,
                checkpoint_every=2,
            )

            expected = []
            for game_index in range(2):
                expected.extend(generate_teacher_episode(
                    player_count=3,
                    game_seed=79 + game_index,
                    teacher_seed=80 + game_index,
                    simulations=1,
                    depth=1,
                    max_rounds=1,
                ))
            self.assertEqual(read_teacher_dataset(output), expected)
            self.assertEqual(state.completed_games, 2)
            self.assertEqual(state.gpu_seconds, 0.0)
            self.assertGreater(state.teacher_cpu_seconds, 0.0)
            self.assertEqual(
                len(list((run_directory / "games").glob("game-*.jsonl"))),
                2,
            )

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

    def test_generate_cli_records_restart_command_and_completed_status(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            output = root / "teacher.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.search_pipeline.generate",
                    "--output",
                    str(output),
                    "--run-dir",
                    str(run_directory),
                    "--players",
                    "3",
                    "--games",
                    "1",
                    "--simulations",
                    "1",
                    "--depth",
                    "1",
                    "--max-rounds",
                    "1",
                    "--workers",
                    "1",
                    "--checkpoint-every",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(
                (run_directory / "search-run.json").read_text(encoding="utf-8"),
            )
            manifest = json.loads(
                (run_directory / "training-command.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["arguments"][:2], ["-m", "training.search_pipeline.generate"])
            self.assertEqual(manifest["pidFile"], "launcher-pid.txt")
            self.assertEqual(manifest["pauseFile"], "pause.request")
            self.assertEqual(manifest["stateFile"], "search-run.json")

    def test_beginning_a_paused_search_run_consumes_the_stale_pause_request(self) -> None:
        with TemporaryDirectory() as directory:
            control = SearchRunControl(Path(directory))
            state = SearchRunState(status="paused", completed_games=7)
            control.save(state)
            control.request_pause()

            resumed = control.begin()
            self.assertEqual(resumed.completed_games, 7)
            self.assertEqual(resumed.status, "running")
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

    def test_league_training_pauses_at_a_safe_boundary_and_resumes_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(184)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save(
                {
                    "version": 1,
                    "stage": "search-teacher-distillation",
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                },
                base,
            )

            def roster() -> LeagueRoster:
                value = LeagueRoster(seed=185)
                value.add("frozen-v1", str(base), frozen=True)
                return value

            run_directory = root / "resumed-run"
            resumed_output = root / "resumed.pt"
            paused = run_league_training(
                checkpoint=base,
                roster=roster(),
                output=resumed_output,
                games=2,
                player_counts=(3,),
                seed=186,
                max_rounds=1,
                device="cpu",
                run_directory=run_directory,
                should_pause=lambda game_index, decisions: game_index >= 1 and decisions == 0,
            )
            self.assertEqual(paused["status"], "paused")
            self.assertFalse(resumed_output.exists())
            self.assertTrue((run_directory / "league-checkpoint.pt").exists())
            self.assertEqual(
                json.loads((run_directory / "league-state.json").read_text())["status"],
                "paused",
            )

            resumed = run_league_training(
                checkpoint=base,
                roster=roster(),
                output=resumed_output,
                games=2,
                player_counts=(3,),
                seed=186,
                max_rounds=1,
                device="cpu",
                run_directory=run_directory,
            )
            uninterrupted_output = root / "uninterrupted.pt"
            uninterrupted = run_league_training(
                checkpoint=base,
                roster=roster(),
                output=uninterrupted_output,
                games=2,
                player_counts=(3,),
                seed=186,
                max_rounds=1,
                device="cpu",
            )

            self.assertEqual(resumed, uninterrupted)
            resumed_payload = torch.load(resumed_output, map_location="cpu", weights_only=False)
            uninterrupted_payload = torch.load(
                uninterrupted_output,
                map_location="cpu",
                weights_only=False,
            )
            for name, expected in uninterrupted_payload["model"].items():
                self.assertTrue(torch.equal(resumed_payload["model"][name], expected), name)
            self.assertEqual(
                json.loads((run_directory / "league-state.json").read_text())["status"],
                "completed",
            )
            self.assertFalse((run_directory / "league-checkpoint.pt").exists())

    def test_league_training_freezes_a_candidate_when_gpu_budget_is_reached(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(187)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save(
                {
                    "version": 1,
                    "stage": "search-teacher-distillation",
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                },
                base,
            )
            roster = LeagueRoster(seed=188)
            roster.add("frozen-v1", str(base), frozen=True)
            run_directory = root / "budgeted-run"
            output = root / "milestone.pt"

            metrics = run_league_training(
                checkpoint=base,
                roster=roster,
                output=output,
                games=3,
                player_counts=(3,),
                seed=189,
                max_rounds=1,
                device="cpu",
                run_directory=run_directory,
                gpu_hours=1e-12,
            )

            self.assertEqual(metrics["status"], "budget-reached")
            self.assertEqual(metrics["completedGames"], 1)
            self.assertGreater(metrics["gpuSeconds"], 0.0)
            self.assertTrue(output.is_file())
            state = json.loads((run_directory / "league-state.json").read_text())
            self.assertEqual(state["status"], "budget-reached")
            self.assertEqual(state["completed_games"], 1)
            self.assertEqual(state["gpu_seconds"], metrics["gpuSeconds"])

    def test_league_cli_records_the_gpu_budget_for_guarded_resume(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(190)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save(
                {
                    "version": 1,
                    "stage": "search-teacher-distillation",
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                },
                base,
            )
            run_directory = root / "run"
            output = root / "milestone.pt"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.search_pipeline.train_league",
                    "--checkpoint",
                    str(base),
                    "--opponent",
                    f"frozen-v1={base}",
                    "--output",
                    str(output),
                    "--run-dir",
                    str(run_directory),
                    "--games",
                    "3",
                    "--players",
                    "3",
                    "--max-rounds",
                    "1",
                    "--device",
                    "cpu",
                    "--gpu-hours",
                    "0.000000000001",
                ],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (run_directory / "training-command.json").read_text(encoding="utf-8"),
            )
            budget_index = manifest["arguments"].index("--gpu-hours")
            self.assertEqual(manifest["arguments"][budget_index + 1], "0.000000000001")
            state = json.loads((run_directory / "league-state.json").read_text())
            self.assertEqual(state["status"], "budget-reached")
            self.assertTrue(output.is_file())

    def test_league_training_updates_in_bounded_game_batches(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(191)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save(
                {
                    "version": 1,
                    "stage": "search-teacher-distillation",
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                },
                base,
            )
            roster = LeagueRoster(seed=192)
            roster.add("frozen-v1", str(base), frozen=True)
            output = root / "league.pt"

            metrics = run_league_training(
                checkpoint=base,
                roster=roster,
                output=output,
                games=25,
                player_counts=(3,),
                seed=193,
                max_rounds=1,
                device="cpu",
            )

            self.assertEqual(metrics["updates"], 2)
            payload = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(payload["metrics"]["updates"], 2)

    def test_league_training_resumes_after_a_periodic_update(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96)
            torch.manual_seed(194)
            model = PropertyTradingPolicy(config)
            base = root / "base.pt"
            torch.save(
                {
                    "version": 1,
                    "stage": "search-teacher-distillation",
                    "model_config": asdict(config),
                    "model": model.state_dict(),
                },
                base,
            )

            def roster() -> LeagueRoster:
                value = LeagueRoster(seed=195)
                value.add("frozen-a", str(base), frozen=True)
                value.add("frozen-b", str(base), frozen=True)
                return value

            run_directory = root / "resumed-run"
            resumed_output = root / "resumed.pt"
            checkpoint_seen_before_pause: list[bool] = []

            def pause_after_periodic_update(game_index: int, decisions: int) -> bool:
                if game_index == 24 and decisions == 0:
                    checkpoint_seen_before_pause.append(
                        (run_directory / "league-checkpoint.pt").is_file(),
                    )
                    return True
                return False

            paused = run_league_training(
                checkpoint=base,
                roster=roster(),
                output=resumed_output,
                games=25,
                player_counts=(3,),
                seed=196,
                max_rounds=1,
                device="cpu",
                run_directory=run_directory,
                should_pause=pause_after_periodic_update,
            )
            self.assertEqual(paused["completedGames"], 24)
            self.assertEqual(checkpoint_seen_before_pause, [True])

            resumed = run_league_training(
                checkpoint=base,
                roster=roster(),
                output=resumed_output,
                games=25,
                player_counts=(3,),
                seed=196,
                max_rounds=1,
                device="cpu",
                run_directory=run_directory,
            )
            uninterrupted_output = root / "uninterrupted.pt"
            uninterrupted = run_league_training(
                checkpoint=base,
                roster=roster(),
                output=uninterrupted_output,
                games=25,
                player_counts=(3,),
                seed=196,
                max_rounds=1,
                device="cpu",
            )

            self.assertEqual(resumed, uninterrupted)
            resumed_payload = torch.load(resumed_output, map_location="cpu", weights_only=False)
            uninterrupted_payload = torch.load(
                uninterrupted_output,
                map_location="cpu",
                weights_only=False,
            )
            for name, expected in uninterrupted_payload["model"].items():
                self.assertTrue(torch.equal(resumed_payload["model"][name], expected), name)

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

    def test_legacy_policy_declines_an_unaffordable_purchase(self) -> None:
        legacy = LegacyNeuroPolicy(Path("app/pretrained-model-3p.json"))
        env = PropertyTradingEnv(3, seed=20260910, max_rounds=11)
        env.actor = 1
        env.active = 1
        env.cash[1] = 0
        env.pending_tile = 1
        env.phase = "buy"

        legal = env.legal_actions()
        self.assertEqual([action.kind for action in legal], [ActionKind.DECLINE])
        self.assertEqual(legacy(public_view(env, 1)), legal[0])

    def test_legacy_policy_uses_an_available_debt_trade(self) -> None:
        legacy = LegacyNeuroPolicy(Path("app/pretrained-model.json"))
        env = PropertyTradingEnv(4, seed=20260911, max_rounds=12)
        env.actor = 3
        env.active = 3
        env.cash[3] = -100
        env.owner[1] = 3
        env.mortgaged[1] = True
        env.phase = "debt"

        legal = env.legal_actions()
        self.assertTrue(legal)
        self.assertEqual({action.kind for action in legal}, {ActionKind.TRADE})
        action = legacy(public_view(env, 3))
        self.assertIn(action, legal)
        self.assertLess(action.cash, 0)

    def test_versioned_training_config_drives_fast_and_upper_bound_policies(self) -> None:
        config = load_search_config()
        self.assertEqual(config["teacher"]["version"], "seeded-candidate-rollout-v2")
        self.assertEqual(config["teacher"]["simulations"], 32)
        self.assertEqual(config["teacher"]["depth"], 24)
        self.assertEqual(config["student"]["version"], "search-student-v1")
        self.assertEqual(
            distillation_defaults(),
            {
                "steps": 20_000,
                "batch_size": 256,
                "checkpoint_every": 100,
            },
        )
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

    def test_teacher_generator_module_starts_without_runpy_warning(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "training.search_pipeline.generate", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("RuntimeWarning", result.stderr)


if __name__ == "__main__":
    unittest.main()
