"""
Serves the bot over HTTP for the Jass server. Run from this directory with:

    flask --app player_service run --host=0.0.0.0 --port=8888

The bot URL is then http://<public-host>:8888/rule_based
"""
import logging

from jass.service.player_service_app import PlayerServiceApp

from rule_based_agent import RuleBasedAgent


def create_app():
    logging.basicConfig(level=logging.INFO)
    app = PlayerServiceApp('player_service')
    app.add_player('rule_based', RuleBasedAgent())
    return app
