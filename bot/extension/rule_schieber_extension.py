"""
Helpers on top of the Schieber rules (jass.game.rule_schieber) used by the rule-based agent.
"""
import numpy as np

from jass.game.const import OBE_ABE, UNE_UFE, J_offset, color_masks, color_of_card, lower_trump
from jass.game.game_observation import GameObservation


def beats(card: int, best: int, trump: int) -> bool:
    """
    True if `card` would take the trick from `best`, the card currently winning it.

    The lead color is not needed as a parameter: `best` is always either of the lead color or a trump,
    so a card of any other color can never win.
    """
    card_color = color_of_card[card]
    best_color = color_of_card[best]

    if trump == OBE_ABE or trump == UNE_UFE:
        # no trump color: only a card of the same color can win
        if card_color != best_color:
            return False
        # card ids go from Ace (low id) to Six (high id) within a color
        return card > best if trump == UNE_UFE else card < best

    if card_color == trump:
        if best_color != trump:
            return True  # any trump beats a non-trump card
        return bool(lower_trump[card, best])  # True if best is a lower trump than card

    if best_color == trump or card_color != best_color:
        return False  # non-trump cannot beat a trump, and a different color cannot win

    return card < best


def card_strength(card: int, trump: int) -> int:
    """
    Rank of `card` within its own color under `trump`, from 0 (weakest) to 8 (strongest).
    Only comparable between cards of the same color (a trump always beats a non-trump, see `beats`).
    """
    if color_of_card[card] == trump:
        return int(lower_trump[card, :].sum())  # number of trumps lower than this card: Jack 8, Nine 7, ...
    offset = card % 9  # 0 = Ace ... 8 = Six
    return offset if trump == UNE_UFE else 8 - offset


def played_cards(obs: GameObservation) -> np.ndarray:
    """
    One-hot array (36) of all cards played so far in this round, including the current trick.
    """
    played = np.zeros(36, dtype=np.int32)
    played[obs.tricks[obs.tricks >= 0]] = 1
    return played


def unseen_cards(obs: GameObservation) -> np.ndarray:
    """
    One-hot array (36) of the cards that are still in the other players' hands: not played and not in my hand.
    """
    return 1 - played_cards(obs) - obs.hand


def is_bock(card: int, unseen: np.ndarray, trump: int) -> bool:
    """
    True if no unseen card of the same color is stronger than `card`, i.e. `card` is the highest one of its color
    still out (cards in my own hand do not count against it). For a non-trump card this ignores that an opponent
    who is void in the color could still beat it with a trump.
    """
    strength = card_strength(card, trump)
    same_color_unseen = np.flatnonzero(unseen * color_masks[color_of_card[card]])
    return all(card_strength(c, trump) < strength for c in same_color_unseen)


def void_colors(obs: GameObservation) -> np.ndarray:
    """
    Boolean array [player, color] (4 x 4): True if the player is known to have no cards of that color left.

    A player who did not follow the lead color is void in it, with two exceptions from the Schieber rules:
    - playing a trump never proves anything, since trumping is always allowed, even when able to follow;
    - when trump is led, the Buur does not have to be played, so not following only proves a trump void
      once the Buur is no longer in another player's hand (played, or in our hand).
    """
    voids = np.zeros((4, 4), dtype=bool)
    trump = obs.trump
    has_trump_color = trump < OBE_ABE
    buur_accounted_for = has_trump_color and not unseen_cards(obs)[trump * 9 + J_offset]

    for t in range(min(obs.nr_tricks + 1, 9)):
        trick = obs.tricks[t]
        if trick[0] == -1:
            break
        lead_color = color_of_card[trick[0]]
        for pos in range(1, 4):
            card = trick[pos]
            if card == -1:
                break
            color = color_of_card[card]
            if color == lead_color or (has_trump_color and color == trump):
                continue
            if has_trump_color and lead_color == trump and not buur_accounted_for:
                continue
            voids[(obs.trick_first_player[t] - pos) % 4, lead_color] = True

    # for ourselves we know the exact answer (the Buur exception above would otherwise mislead us about our own hand)
    if obs.player_view != -1:
        voids[obs.player_view] = (color_masks * obs.hand).sum(axis=1) == 0
    return voids


def can_be_trumped(card: int, trump: int, unseen: np.ndarray, voids: np.ndarray, opponents) -> bool:
    """
    True if one of `opponents` (players still to play after us) is known to be void in the color of the non-trump
    `card` and may still hold a trump, so they could take it with a trump.
    """
    color = color_of_card[card]
    if trump >= OBE_ABE or color == trump or not (unseen * color_masks[trump]).any():
        return False
    return any(voids[p, color] and not voids[p, trump] for p in opponents)


def winning_position(trick: np.ndarray, trump: int) -> int:
    """
    Position (0..3) within `trick` of the card currently winning it. Works for incomplete tricks:
    cards not played yet are -1 and are ignored. The trick must contain at least one card.

    The winning card is trick[pos], the player who played it is (first_player - pos) % 4.
    """
    best_pos = 0
    for pos in range(1, 4):
        card = trick[pos]
        if card == -1:
            break  # cards are played in order, so the rest of the trick is empty
        if beats(card, trick[best_pos], trump):
            best_pos = pos
    return best_pos
