"""
Run from this directory with: python -m unittest
"""
import unittest

import numpy as np
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
from jass.game.const import CLUBS, OBE_ABE, PUSH, UNE_UFE, card_ids
from jass.game.game_observation import GameObservation

from player_service import create_app
from agent_rule_based_v2 import RuleBasedAgent

# example messages from the interface definition (JassInterfaceV02.pdf)
TRUMP_REQUEST = {
    "version": "V0.2", "dealer": 0, "currentPlayer": 3, "playerView": 3, "forehand": -1, "trump": -1, "tricks": [],
    "jassTyp": "SCHIEBER_1000",
    "player": [{"hand": []}, {"hand": []}, {"hand": []},
               {"hand": ["D10", "HA", "HJ", "SQ", "S10", "S7", "S6", "CK", "C7"]}]}
TRUMP_REQUEST_AFTER_PUSH = {
    "version": "V0.2", "dealer": 0, "currentPlayer": 1, "playerView": 1, "forehand": 0, "trump": -1, "tricks": [],
    "jassTyp": "SCHIEBER_1000",
    "player": [{"hand": []}, {"hand": ["DJ", "D9", "H6", "SK", "CA", "CJ", "C9", "C8", "C6"]}, {"hand": []},
               {"hand": []}]}
CARD_REQUEST = {
    "version": "V0.2", "dealer": 1, "currentPlayer": 1, "playerView": 1, "trump": 4, "forehand": 0,
    "tricks": [{"cards": ["SQ", "S9", "S10", "SJ"], "points": 15, "win": 0, "first": 0},
               {"cards": ["DQ", "DA", "D8"], "first": 0}],
    "jassTyp": "SCHIEBER_1000",
    "player": [{"hand": []}, {"hand": ["D7", "D6", "HA", "HK", "H9", "H7", "H6", "C8"]}, {"hand": []}, {"hand": []}]}


def observation(hand, forehand):
    obs = GameObservation()
    obs.hand = np.zeros(36, np.int32)
    obs.hand[[card_ids[card] for card in hand]] = 1
    obs.forehand = forehand
    return obs


class TestRules(unittest.TestCase):
    def test_forehand_pushes_weak_hand(self):
        hand = ['D10', 'HA', 'HJ', 'SQ', 'S10', 'S7', 'S6', 'CK', 'C7']  # best score 54 < PUSH_THRESHOLD
        self.assertEqual(RuleBasedAgent().action_trump(observation(hand, forehand=-1)), PUSH)

    def test_forehand_selects_strong_trump_color(self):
        hand = ['DA', 'H7', 'S6', 'CA', 'CK', 'CQ', 'CJ', 'C9', 'C8']  # Buur and Nell of clubs
        self.assertEqual(RuleBasedAgent().action_trump(observation(hand, forehand=-1)), CLUBS)

    def test_forehand_selects_obe_abe_with_aces(self):
        hand = ['DA', 'DK', 'DQ', 'HA', 'HK', 'HQ', 'SA', 'SK', 'CA']
        self.assertEqual(RuleBasedAgent().action_trump(observation(hand, forehand=-1)), OBE_ABE)

    def test_forehand_selects_une_ufe_with_low_cards(self):
        hand = ['D6', 'D7', 'D8', 'H6', 'H7', 'S6', 'S8', 'C6', 'C7']
        self.assertEqual(RuleBasedAgent().action_trump(observation(hand, forehand=-1)), UNE_UFE)

    def test_partner_must_not_push(self):
        hand = ['D10', 'HA', 'HJ', 'SQ', 'S10', 'S7', 'S6', 'CK', 'C7']  # weak hand that forehand would push
        self.assertNotEqual(RuleBasedAgent().action_trump(observation(hand, forehand=0)), PUSH)

    def test_full_games_are_legal(self):
        # the arena raises on an illegal trump or card
        arena = Arena(nr_games_to_play=200, check_move_validity=True)
        bot, opponent = RuleBasedAgent(), AgentRandomSchieber()
        arena.set_players(bot, opponent, bot, opponent)
        arena.play_all_games()
        self.assertEqual(arena.nr_games_played, 200)


class TestService(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_player_is_reachable(self):
        self.assertEqual(self.client.get('/rule_based').status_code, 200)

    def test_trump_requests(self):
        response = self.client.post('/rule_based/action_trump', json=TRUMP_REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['trump'], PUSH)  # weak hand, see test_forehand_pushes_weak_hand

        response = self.client.post('/rule_based/action_trump', json=TRUMP_REQUEST_AFTER_PUSH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['trump'], CLUBS)

    def test_card_request_returns_legal_card(self):
        response = self.client.post('/rule_based/action_play_card', json=CARD_REQUEST)
        self.assertEqual(response.status_code, 200)
        # diamonds were led and D7 and D6 are the only diamonds in hand; under Obe-Abe both are worth 0
        self.assertIn(response.get_json()['card'], ['D7', 'D6'])

    def test_game_info_is_accepted(self):
        self.assertEqual(self.client.post('/rule_based/game_info', json=CARD_REQUEST).status_code, 200)


if __name__ == '__main__':
    unittest.main()
