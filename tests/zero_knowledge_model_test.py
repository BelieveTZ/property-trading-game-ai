from __future__ import annotations

import unittest

import torch

from training.zero_knowledge.env import MonopolyEnv
from training.zero_knowledge.encoding import (
    ACTION_NUMERIC_FEATURES,
    GLOBAL_FEATURES,
    PLAYER_FEATURES,
    PROPERTY_FEATURES,
    collate_action_feature_rows,
)
from training.zero_knowledge.model import ModelConfig, MonopolyPolicy, collate_actions, collate_observations


class ZeroKnowledgeModelTests(unittest.TestCase):
    def test_live_and_replay_action_batches_share_one_checkpoint_compatible_schema(self) -> None:
        envs = [MonopolyEnv(3, seed=11, max_rounds=2), MonopolyEnv(5, seed=12, max_rounds=2)]
        action_lists = [env.legal_actions() for env in envs]
        live_batch, live_mask = collate_actions(
            action_lists,
            [env.actor for env in envs],
            [env.player_count for env in envs],
            "cpu",
        )
        replay_batch, replay_mask = collate_action_feature_rows(
            [env.action_features(actions, env.actor, env.player_count) for env, actions in zip(envs, action_lists, strict=True)],
            "cpu",
        )

        self.assertEqual((GLOBAL_FEATURES, PLAYER_FEATURES, PROPERTY_FEATURES, ACTION_NUMERIC_FEATURES), (37, 11, 17, 7))
        self.assertTrue(torch.equal(live_mask, replay_mask))
        for key in live_batch:
            self.assertTrue(torch.equal(live_batch[key], replay_batch[key]), key)

    def test_policy_scores_only_legal_candidates_and_backpropagates(self) -> None:
        envs = [MonopolyEnv(3, seed=1, max_rounds=2), MonopolyEnv(5, seed=2, max_rounds=2)]
        observations = [env.observe() for env in envs]
        action_lists = [env.legal_actions() for env in envs]
        model = MonopolyPolicy(ModelConfig(d_model=48, nhead=4, layers=1, feedforward=96, dropout=0.0))
        observation_batch = collate_observations(observations, "cpu")
        action_batch, mask = collate_actions(
            action_lists,
            [env.actor for env in envs],
            [env.player_count for env in envs],
            "cpu",
        )
        logits, values, hidden = model(observation_batch, action_batch, mask)
        self.assertEqual(logits.shape, mask.shape)
        self.assertEqual(values.shape, (2,))
        self.assertEqual(hidden.shape, (2, 48))
        self.assertTrue(torch.isfinite(logits[mask]).all())
        loss = -torch.log_softmax(logits, dim=-1)[0, 0] + values.square().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
