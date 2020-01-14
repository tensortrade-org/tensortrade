
import pytest
import unittest.mock as mock

from tensortrade import TradingContext
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import StochasticExchange
from tensortrade.instruments import USD, BTC, ETH
from tensortrade.wallets import Portfolio
from tensortrade.actions import ManagedRiskOrders, DynamicOrders


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


def test_init():

    # Test 1
    exchange = StochasticExchange()
    portfolio = Portfolio(USD, wallets=[
        (exchange, USD, 10000),
        (exchange, BTC, 0)
    ])

    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme="managed-risk",
        reward_scheme="simple"
    )

    assert env

    # Test 2
    portfolio = Portfolio(USD, wallets=[
        (exchange, USD, 10000),
        (exchange, BTC, 0)
    ])

    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme="dynamic",
        reward_scheme="simple"
    )

    assert env

    # Test 3
    action_scheme = ManagedRiskOrders() + DynamicOrders()

    portfolio = Portfolio(USD, wallets=[
        (exchange, USD, 10000),
        (exchange, BTC, 0)
    ])

    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme="simple"
    )

    assert env

    pytest.fail("Failed.")
