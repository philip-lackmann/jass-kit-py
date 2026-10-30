"""
Simple rule-based Jass agent (Exercise 2). It is the baseline for everything built later.
"""
import numpy as np

from jass.agents.agent import Agent
from jass.game.const import PUSH, card_values, color_masks
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

# forehand pushes unless it holds at least this many cards of one color
MIN_CARDS_FOR_TRUMP = 4


class RuleBasedAgent(Agent):
    def __init__(self):
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Select the color with the most cards. Only forehand may push (forehand is -1 until it has decided),
        a player that received a push has to select a trump.
        """
        cards_per_color = (color_masks * obs.hand).sum(axis=1)
        if obs.forehand == -1 and cards_per_color.max() < MIN_CARDS_FOR_TRUMP:
            return PUSH
        return int(cards_per_color.argmax())

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Play the valid card that is worth the most points under the current trump.
        """
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))
        return int(valid_cards[card_values[obs.trump, valid_cards].argmax()])
