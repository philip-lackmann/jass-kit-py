"""
Serves the bot over HTTP for the Jass server. Requires Python 3.12 or later. Run from this directory with:

    pip install -r requirements.txt
    flask --app player_service run --host=0.0.0.0 --port=8888

In production (Linux only), deploy/jass-bot.service runs it with gunicorn instead:

    gunicorn 'player_service:create_app()' --bind 0.0.0.0:8888 --workers 2 --timeout 30 --access-logfile -

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
