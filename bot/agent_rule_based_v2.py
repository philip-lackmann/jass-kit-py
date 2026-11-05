"""
Simple rule-based Jass agent (Exercise 2). It is the baseline for everything built later.
"""
import numpy as np

from jass.agents.agent import Agent
from jass.game.const import PUSH, OBE_ABE, UNE_UFE, card_values, color_masks, color_of_card, next_player, \
    partner_player
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

from extension.rule_schieber_extension import beats, can_be_trumped, card_strength, is_bock, unseen_cards, \
    void_colors, winning_position

# trump selection: score of each card by its offset in the color (A, K, Q, J, 10, 9, 8, 7, 6)
TRUMP_SCORE = np.array([15, 10, 7, 25, 6, 19, 5, 5, 5])      # card of the trump color (J = Buur, 9 = Nell)
NO_TRUMP_SCORE = np.array([9, 7, 5, 2, 1, 0, 0, 0, 0])       # card of another color, when a color is trump
OBE_ABE_SCORE = np.array([14, 10, 8, 7, 5, 0, 5, 0, 0])
UNE_UFE_SCORE = np.array([0, 2, 1, 1, 5, 5, 7, 9, 11])


def trump_scores(hand: np.ndarray) -> np.ndarray:
    """
    Score of the hand for each of the 6 trump modes (DIAMONDS..CLUBS, OBE_ABE, UNE_UFE), higher is better.
    """
    hand_by_color = hand.reshape(4, 9)  # [color, offset]
    scores = np.zeros(6)
    for trump in range(4):
        for color in range(4):
            table = TRUMP_SCORE if color == trump else NO_TRUMP_SCORE
            scores[trump] += (hand_by_color[color] * table).sum()
    scores[OBE_ABE] = (hand_by_color * OBE_ABE_SCORE).sum()
    scores[UNE_UFE] = (hand_by_color * UNE_UFE_SCORE).sum()
    return scores


class RuleBasedAgentV2(Agent):
    # forehand pushes if no trump mode scores at least this much
    PUSH_THRESHOLD = 68
    # a trump is only used to win a trick if the cards already in it are worth at least this many points
    MIN_TRICK_POINTS_FOR_TRUMP = 10

    def __init__(self):
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Select the trump mode with the best hand score. Only forehand may push (forehand is -1 until it has
        decided), a player that received a push has to select a trump.
        """
        scores = trump_scores(obs.hand)
        if obs.forehand == -1 and scores.max() < self.PUSH_THRESHOLD:
            return PUSH
        return int(scores.argmax())

    def action_play_card(self, obs: GameObservation) -> int:
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))
        unseen = unseen_cards(obs)
        voids = void_colors(obs)

        # players still to play in this trick after us, and the opponents among them
        after_us = []
        player = obs.player
        for _ in range(3 - obs.nr_cards_in_trick):
            player = next_player[player]
            after_us.append(player)
        opponents_after_us = [p for p in after_us if p != partner_player[obs.player]]

        # first to play in trick
        if obs.nr_cards_in_trick == 0:
            our_team_declared = obs.declared_trump in (obs.player, partner_player[obs.player])
            return self.lead(valid_cards, obs.trump, unseen, voids, opponents_after_us, our_team_declared)

        winning_pos = winning_position(obs.current_trick, obs.trump)
        best_card = obs.current_trick[winning_pos]
        best_player = (obs.trick_first_player[obs.nr_tricks] - winning_pos) % 4
        is_partner_winning = best_player == partner_player[obs.player]
        trick_points = int(card_values[obs.trump, obs.current_trick[:obs.nr_cards_in_trick]].sum())

        # last to play in trick, easiest move to determine
        if obs.nr_cards_in_trick == 3:
            if is_partner_winning:
                # schmieren
                return self.smear(valid_cards, obs.trump, unseen)
            else:
                return self.play_last_against_opponent(valid_cards, best_card, obs.trump, unseen, trick_points)

        # 2nd or 3rd to play: in both cases exactly one opponent still plays after us
        return self.play_middle(valid_cards, best_card, obs.trump, unseen, voids, opponents_after_us,
                                is_partner_winning, trick_points)

    def lead(self, valid_cards: np.ndarray, trump: int, unseen: np.ndarray, voids: np.ndarray, opponents,
             our_team_declared: bool) -> int:
        """
        First to play in the trick.
        - Our team chose trump and opponents may still have trumps: lead a Bock trump to pull them (Trumpf ziehen).
        - Otherwise lead a non-trump Bock that no opponent is known to be able to trump, in the color with the most
          unseen cards (least likely that someone is void). While trumps are out, skip colors without unseen cards.
        - No Bock: lead a small card (same order as discard).
        """
        has_trump_color = trump < OBE_ABE
        unseen_trumps = (unseen * color_masks[trump]).sum() if has_trump_color else 0
        unseen_per_color = (color_masks * unseen).sum(axis=1)
        bocks = [c for c in valid_cards if is_bock(c, unseen, trump)]

        if our_team_declared and unseen_trumps > 0 and any(not voids[p, trump] for p in opponents):
            trump_bocks = [c for c in bocks if color_of_card[c] == trump]
            if trump_bocks:
                return int(min(trump_bocks, key=lambda c: card_strength(c, trump)))  # keep the higher ones

        side_bocks = [c for c in bocks if color_of_card[c] != trump
                      and not can_be_trumped(c, trump, unseen, voids, opponents)]
        if unseen_trumps > 0:
            side_bocks = [c for c in side_bocks if unseen_per_color[color_of_card[c]] > 0]
        if side_bocks:
            return int(max(side_bocks, key=lambda c: (unseen_per_color[color_of_card[c]], card_values[trump, c])))

        return self.discard(valid_cards, trump)

    def play_middle(self, valid_cards: np.ndarray, best_card: int, trump: int, unseen: np.ndarray,
                    voids: np.ndarray, opponents, is_partner_winning: bool, trick_points: int) -> int:
        """
        2nd or 3rd to play. A card is safe if it is Bock and the opponent after us is not known to be able to trump
        it (void in its color and possibly holding trumps).
        - Partner is winning with a safe card: smear.
        - We have a safe card that wins: play it (non-trump with the most points, else the weakest trump if the
          trick is worth a trump). This also takes over from the partner if their card is not safe.
        - No safe winner and an opponent is winning: still try to win, choosing the card like the last player would
          (measured ~0.6 points per game better than discarding).
        - Otherwise (partner winning, but not safe, and we cannot secure it): discard.
        """
        def is_safe(card):
            return is_bock(card, unseen, trump) and not can_be_trumped(card, trump, unseen, voids, opponents)

        if is_partner_winning and is_safe(best_card):
            return self.smear(valid_cards, trump, unseen)

        safe_winners = [c for c in valid_cards if beats(c, best_card, trump) and is_safe(c)]
        if safe_winners:
            non_trump = [c for c in safe_winners if color_of_card[c] != trump]
            if non_trump:
                return int(max(non_trump, key=lambda c: (card_values[trump, c], -card_strength(c, trump))))
            if trick_points >= self.MIN_TRICK_POINTS_FOR_TRUMP:
                return int(min(safe_winners, key=lambda c: card_strength(c, trump)))

        if not is_partner_winning:
            return self.play_last_against_opponent(valid_cards, best_card, trump, unseen, trick_points)
        return self.discard(valid_cards, trump)

    def play_last_against_opponent(self, valid_cards: np.ndarray, best_card: int, trump: int, unseen: np.ndarray,
                                   trick_points: int) -> int:
        """
        Last to play and an opponent is winning the trick (also used by play_middle as a fallback).
        - If a non-trump card wins: play the one with the most points, but keep Bock cards (highest of their
          color still out) if another card also wins. As the last player its points are safe for us.
        - If only trumps win: play the weakest of them, to keep Buur/Nell for later, but only if the trick is
          worth at least MIN_TRICK_POINTS_FOR_TRUMP.
        - Otherwise: throw away the card with the fewest points, preferring non-trumps and weak cards.
        """
        winners = [c for c in valid_cards if beats(c, best_card, trump)]
        if winners:
            non_trump = [c for c in winners if color_of_card[c] != trump]
            if non_trump:
                not_bock = [c for c in non_trump if not is_bock(c, unseen, trump)] or non_trump
                return int(max(not_bock, key=lambda c: (card_values[trump, c], -card_strength(c, trump))))
            if trick_points >= self.MIN_TRICK_POINTS_FOR_TRUMP:
                return int(min(winners, key=lambda c: card_strength(c, trump)))

        return self.discard(valid_cards, trump)

    @staticmethod
    def smear(valid_cards: np.ndarray, trump: int, unseen: np.ndarray) -> int:
        """
        The partner is winning the trick safely, so every point we add goes to our team.
        Give the non-trump, non-Bock card with the most points: it would probably be lost later anyway
        (e.g. a Ten while the Ace is still out). If there is none, keep Bocks and trumps and discard instead.
        """
        candidates = [c for c in valid_cards if color_of_card[c] != trump and not is_bock(c, unseen, trump)]
        if candidates:
            return int(max(candidates, key=lambda c: (card_values[trump, c], -card_strength(c, trump))))
        return RuleBasedAgentV2.discard(valid_cards, trump)

    @staticmethod
    def discard(valid_cards: np.ndarray, trump: int) -> int:
        """
        Throw away the least useful card (abwerfen): non-trumps first, then fewest points, then weakest.
        """
        return int(min(valid_cards, key=lambda c: (color_of_card[c] == trump,
                                                   card_values[trump, c],
                                                   card_strength(c, trump))))
