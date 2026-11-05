"""
Plays the bot against the random agent locally and reports the mean points with a 95% confidence interval.
"""
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

from agent_rule_based_v2 import RuleBasedAgentV2

NR_GAMES = 1000
POINTS_TO_WIN = 157 / 2


def main():
    arena = Arena(nr_games_to_play=NR_GAMES)
    bot = RuleBasedAgentV2()
    opponent = AgentRandomSchieber()
    arena.set_players(bot, opponent, bot, opponent)
    arena.play_all_games()

    points = arena.points_team_0
    margin = 1.96 * points.std(ddof=1) / NR_GAMES ** 0.5
    print('Bot points per game: {:.1f} +/- {:.1f} (of 157)'.format(points.mean(), margin))
    print('Games won by bot: {:.1%}'.format((points > POINTS_TO_WIN).mean()))


if __name__ == '__main__':
    main()
