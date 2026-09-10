from __future__ import annotations

import unittest

import numpy as np

from training.zero_knowledge.board_rules import board_rule_for, indexed_board_rules
from training.zero_knowledge.encoding import encode_observation
from training.zero_knowledge.env import ActionKind, DEED_TILES, MonopolyEnv


class ZeroKnowledgeEnvironmentTests(unittest.TestCase):
    def test_observation_encoding_is_owned_by_the_checkpoint_schema_module(self) -> None:
        env = MonopolyEnv(4, seed=73, max_rounds=4)
        expected = env.observe()
        actual = encode_observation(env)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_python_training_reads_the_canonical_board_rules(self) -> None:
        rules = indexed_board_rules()

        self.assertEqual(board_rule_for(39)["rents"], (50, 200, 600, 1400, 1700, 2000))
        self.assertEqual(int(rules["prices"][9]), 120)
        self.assertEqual(int(rules["mortgages"][28]), 75)
        self.assertEqual(rules["groups"]["sky"], (6, 8, 9))

    def test_third_failed_jail_roll_resolves_landing_after_debt_is_paid(self) -> None:
        class FixedRoll:
            def __init__(self) -> None:
                self.values = iter((1, 2))

            def integers(self, *_args: object, **_kwargs: object) -> int:
                return next(self.values)

        env = MonopolyEnv(3, seed=3, max_rounds=3)
        env.rng = FixedRoll()
        env.active = 0
        env.actor = 0
        env.phase = "jail"
        env.in_jail[0] = True
        env.jail_turns[0] = 2
        env.cash[0] = 20
        env.position[0] = 0
        env.owner[5] = 0

        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.ROLL_JAIL))
        self.assertEqual(env.phase, "debt")
        self.assertEqual(int(env.position[0]), 3)

        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.MORTGAGE and action.tile == 5))
        self.assertEqual(env.phase, "buy")
        self.assertEqual(env.pending_tile, 3)

    def test_trade_receiver_pays_interest_on_mortgaged_deed(self) -> None:
        env = MonopolyEnv(3, seed=5, max_rounds=3)
        env.owner[3] = 1
        env.mortgaged[3] = True
        env.active = 0
        env.actor = 0
        env.phase = "turn_start"

        offer = next(
            action
            for action in env.legal_actions()
            if action.kind == ActionKind.TRADE
            and action.target == 1
            and action.cash == 60
            and action.take_tile == 3
        )
        env.step(offer)
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.ACCEPT_TRADE))

        self.assertEqual(int(env.owner[3]), 0)
        self.assertEqual(int(env.cash[0]), 1437)
        self.assertEqual(int(env.cash[1]), 1560)

    def test_get_out_of_jail_card_leaves_deck_until_used(self) -> None:
        class FixedRoll:
            def __init__(self, values: tuple[int, ...]) -> None:
                self.values = iter(values)

            def integers(self, *_args: object, **_kwargs: object) -> int:
                return next(self.values)

        env = MonopolyEnv(3, seed=7, max_rounds=3)
        env.chance_deck = [8, *[card for card in range(16) if card != 8]]
        env.chance_cursor = 0
        env.rng = FixedRoll((3, 4))
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.ROLL))

        self.assertEqual(int(env.jail_cards[0]), 1)
        self.assertNotIn(8, env.chance_deck)
        self.assertEqual(len(env.chance_deck), 15)

        env.phase = "jail"
        env.actor = 0
        env.active = 0
        env.in_jail[0] = True
        env.rng = FixedRoll((1, 2))
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.USE_JAIL_CARD))

        self.assertEqual(int(env.jail_cards[0]), 0)
        self.assertIn(8, env.chance_deck)
        self.assertEqual(len(env.chance_deck), 16)

    def test_traded_get_out_of_jail_card_keeps_its_deck_identity(self) -> None:
        env = MonopolyEnv(3, seed=9, max_rounds=3)
        env.chance_deck.remove(8)
        env.jail_cards[0] = 1
        env.jail_card_decks[0] = [True]
        offer = next(
            action
            for action in env.legal_actions()
            if action.kind == ActionKind.TRADE
            and action.target == 1
            and action.give_card == 1
        )

        env.step(offer)
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.ACCEPT_TRADE))

        self.assertEqual(env.jail_card_decks[0], [])
        self.assertEqual(env.jail_card_decks[1], [True])

    def test_bankruptcy_to_bank_returns_held_jail_cards_to_decks(self) -> None:
        env = MonopolyEnv(3, seed=11, max_rounds=3)
        env.chance_deck.remove(8)
        env.community_deck.remove(4)
        env.jail_cards[0] = 2
        env.jail_card_decks[0] = [True, False]
        env.cash[0] = -1
        env.phase = "debt"
        env.actor = 0
        env.active = 0
        env.trades_left = 0

        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.BANKRUPT))

        self.assertEqual(int(env.jail_cards[0]), 0)
        self.assertEqual(env.jail_card_decks[0], [])
        self.assertIn(8, env.chance_deck)
        self.assertIn(4, env.community_deck)

    def test_auction_does_not_force_a_winner_after_eighty_actions(self) -> None:
        env = MonopolyEnv(3, seed=13, max_rounds=3)
        env._start_auction(1)
        assert env.auction is not None
        env.auction.update({"bid": 100, "leader": 0, "active": [0, 1, 2], "cursor": 1, "steps": 79})
        env.actor = 1

        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.BID and action.cash == 110))

        self.assertEqual(env.phase, "auction")
        self.assertIsNotNone(env.auction)
        self.assertEqual(env.actor, 2)

    def test_random_play_reaches_terminal_for_each_supported_player_count(self) -> None:
        for players in (3, 4, 5):
            env = MonopolyEnv(players, seed=players, max_rounds=4)
            steps = 0
            while not env.done and steps < 10_000:
                actions = env.legal_actions()
                self.assertTrue(actions)
                env.step(actions[int(env.rng.integers(len(actions)))])
                env.validate_state()
                steps += 1
            self.assertTrue(env.done)
            self.assertAlmostEqual(float(env.terminal_rewards.sum()), 0.0, places=5)
            self.assertAlmostEqual(float(env.terminal_rewards.max()), 1.0, places=5)
            self.assertAlmostEqual(float(env.terminal_rewards.min()), -1.0, places=5)

    def test_only_active_player_can_initiate_trade(self) -> None:
        env = MonopolyEnv(4, seed=7, max_rounds=3)
        env.owner[1] = 0
        env.owner[3] = 1
        env.phase = "turn_start"
        env.active = 0
        env.actor = 0
        trades = [action for action in env.legal_actions() if action.kind == ActionKind.TRADE]
        self.assertTrue(trades)
        offer = trades[0]
        env.step(offer)
        self.assertEqual(env.phase, "trade_response")
        self.assertEqual(env.actor, offer.target)
        self.assertEqual({action.kind for action in env.legal_actions()}, {ActionKind.ACCEPT_TRADE, ActionKind.REJECT_TRADE})

    def test_rejected_offer_is_not_repeated_but_new_offer_is_available(self) -> None:
        env = MonopolyEnv(3, seed=11, max_rounds=3)
        env.owner[1] = 0
        env.owner[3] = 1
        env.phase = "turn_start"
        first = next(action for action in env.legal_actions() if action.kind == ActionKind.TRADE)
        env.step(first)
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.REJECT_TRADE))
        keys = {action.key() for action in env.legal_actions()}
        self.assertNotIn(first.key(), keys)
        self.assertTrue(any(action.kind == ActionKind.TRADE for action in env.legal_actions()))

    def test_observation_and_action_shapes_are_stable(self) -> None:
        env = MonopolyEnv(5, seed=19, max_rounds=3)
        observation = env.observe()
        self.assertEqual(observation["global"].shape, (37,))
        self.assertEqual(observation["players"].shape, (5, 11))
        self.assertEqual(observation["properties"].shape, (len(DEED_TILES), 17))
        actions = env.legal_actions()
        features = env.action_features(actions, env.actor, env.player_count)
        self.assertEqual(features["numeric"].shape, (len(actions), 7))
        self.assertTrue(np.isfinite(observation["properties"]).all())

    def test_multiple_even_builds_and_debt_sales_are_available_in_one_turn(self) -> None:
        env = MonopolyEnv(3, seed=21, max_rounds=3)
        for tile in (6, 8, 9):
            env.owner[tile] = 0
        for tile in (6, 8, 9, 6):
            action = next(item for item in env.legal_actions() if item.kind == ActionKind.BUILD and item.tile == tile)
            env.step(action)
        self.assertEqual(tuple(int(env.houses[tile]) for tile in (6, 8, 9)), (2, 1, 1))
        env.built_tiles.clear()
        env.cash[0] = -200
        env.phase = "debt"
        env.trades_left = 0
        first_sale = next(item for item in env.legal_actions() if item.kind == ActionKind.SELL_BUILDING and item.tile == 6)
        env.step(first_sale)
        second_sale = next(item for item in env.legal_actions() if item.kind == ActionKind.SELL_BUILDING and item.tile in (8, 9))
        env.step(second_sale)
        self.assertEqual(int(env.houses.sum()), 2)

    def test_trade_response_observation_contains_offer_terms(self) -> None:
        env = MonopolyEnv(3, seed=23, max_rounds=3)
        env.owner[1] = 0
        env.owner[3] = 1
        offer = next(
            action
            for action in env.legal_actions()
            if action.kind == ActionKind.TRADE and action.cash > 0 and action.take_tile == 3
        )
        env.step(offer)
        observation = env.observe()["global"]
        self.assertEqual(observation[35], 1.0)
        self.assertAlmostEqual(float(observation[28]), offer.cash / 1500.0)
        self.assertAlmostEqual(float(observation[30]), 3 / 39.0)

    def test_bankruptcy_to_bank_auctions_each_deed(self) -> None:
        env = MonopolyEnv(3, seed=29, max_rounds=3)
        env.owner[1] = 0
        env.owner[3] = 0
        env.cash[0] = -1
        env.phase = "debt"
        env.actor = 0
        env.active = 0
        env.managed_tiles.update((1, 3))
        env.trades_left = 0
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.BANKRUPT))
        self.assertEqual(env.phase, "auction")
        self.assertEqual(env.pending_tile, 1)
        self.assertEqual(env.auction_queue, [3])

    def test_even_building_and_mortgage_rules(self) -> None:
        env = MonopolyEnv(3, seed=31, max_rounds=3)
        for tile in (6, 8, 9):
            env.owner[tile] = 0
        builds = {action.tile for action in env.legal_actions() if action.kind == ActionKind.BUILD}
        self.assertTrue({6, 8, 9}.issubset(builds))
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.BUILD and action.tile == 6))
        builds = {action.tile for action in env.legal_actions() if action.kind == ActionKind.BUILD}
        self.assertNotIn(6, builds)
        self.assertTrue({8, 9}.issubset(builds))
        mortgages = {action.tile for action in env.legal_actions() if action.kind == ActionKind.MORTGAGE}
        self.assertTrue(all(tile not in mortgages for tile in (6, 8, 9)))

    def test_printed_rent_table_and_monopoly_double_rent(self) -> None:
        env = MonopolyEnv(3, seed=37, max_rounds=3)
        env.owner[1] = 0
        self.assertEqual(env._rent(1, 7), 2)
        env.owner[3] = 0
        self.assertEqual(env._rent(1, 7), 4)
        env.houses[1] = 3
        self.assertEqual(env._rent(1, 7), 90)

    def test_uncollectible_rent_does_not_create_cash(self) -> None:
        env = MonopolyEnv(3, seed=41, max_rounds=3)
        env.owner[39] = 1
        env.houses[39] = 5
        env.cash[0] = 100
        env.position[0] = 39
        env.last_roll = 7
        env._resolve_landing(0, no_extra_turn=False)
        self.assertEqual(env.phase, "debt")
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.BANKRUPT))
        self.assertEqual(int(env.cash[1]), 1600)

    def test_collect_from_each_resolves_debtors_in_sequence(self) -> None:
        env = MonopolyEnv(3, seed=43, max_rounds=3)
        env.cash[1] = 5
        env.cash[2] = 20
        self.assertTrue(env._collect_each(0, 10))
        self.assertEqual(env.actor, 1)
        env.step(next(action for action in env.legal_actions() if action.kind == ActionKind.BANKRUPT))
        self.assertFalse(env.alive[1])
        self.assertEqual(int(env.cash[0]), 1515)
        self.assertEqual(int(env.cash[2]), 10)
        self.assertEqual(env.actor, 0)
        self.assertEqual(env.phase, "post_turn")


if __name__ == "__main__":
    unittest.main()
