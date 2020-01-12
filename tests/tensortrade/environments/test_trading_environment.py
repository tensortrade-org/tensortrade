
import unittest.mock as mock

from tensortrade import TradingContext
from tensortrade.environments import TradingEnvironment
from tensortrade.instruments import USD, BTC, ETH


config = {
    'base_instrument': 50,
    'actions': {
        'pairs': [USD/BTC, USD/ETH]
    }
}


def make_env(exchange: str, action: str, reward: str):
    portfolio = mock.Mock()
    return TradingEnvironment(exchange=exchange, action_scheme=action, reward_scheme=reward, portfolio=portfolio)


def test_injects_simulated_discrete_simple_environment():
    env = make_env('simulated', 'dynamic', 'simple')

    assert env.action_scheme.pairs == [USD/BTC]

    with TradingContext(**config):
        env = make_env('simulated', 'dynamic', 'simple')
        assert env.action_scheme.pairs == [USD/BTC, USD/ETH]
