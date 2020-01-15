
import pytest
import unittest.mock as mock
import numpy as np
import pandas as pd

from tensortrade import TradingContext
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.instruments import USD, BTC, ETH
from tensortrade.wallets import Portfolio
from tensortrade.actions import ManagedRiskOrders, DynamicOrders
from tensortrade.data import DataFeed, PortfolioDataSource, DataFrame


config = {
    'base_instrument': 50,
    'actions': {
        'pairs': [USD/BTC, USD/ETH]
    }
}


@pytest.fixture
def exchange_ds():
    data = np.array([
        [13863.13, 13889., 12952.5, 13480.01, 11484.01],
        [13480.01, 15275., 13005., 14781.51, 23957.87],
        [14781.51, 15400., 14628., 15098.14, 16584.63],
        [15098.14, 15400., 14230., 15144.99, 17980.39],
        [15144.99, 17178., 14824.05, 16960.01, 20781.65]
    ])
    index = pd.Index(
        ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
        name="date"
    )
    columns = ["open", "high", "low", "close", "volume"]
    data_frame = pd.DataFrame(data, index=index, columns=columns)

    return DataFrame('exchange', data_frame)


@pytest.fixture
def exchange(exchange_ds):

    exchange = SimulatedExchange(source=exchange_ds,
                                 extract=lambda x: {USD/BTC: x['close']})

    return exchange


@pytest.fixture
def portfolio(exchange):
    portfolio = Portfolio(USD, wallets=[
        (exchange, USD, 10000),
        (exchange, ETH, 200),
        (exchange, BTC, 10)
    ])
    return portfolio


def make_env(action: str, reward: str):
    portfolio = mock.Mock()
    feed = mock.Mock()
    return TradingEnvironment(portfolio=portfolio,
                              action_scheme=action,
                              reward_scheme=reward,
                              feed=feed)


@pytest.fixture
def feed(portfolio, exchange_ds):

    portfolio_ds = PortfolioDataSource(portfolio)

    data_feed = DataFeed(sources=[
        exchange_ds,
        portfolio_ds
    ])
    return data_feed


def test_injects_dynamic_simple_environment(portfolio, feed):
    env = TradingEnvironment(portfolio=portfolio,
                             action_scheme='dynamic',
                             reward_scheme='simple',
                             feed=feed)

    assert not hasattr(env.action_scheme, 'pairs')

    with TradingContext(**config):
        env = TradingEnvironment(portfolio=portfolio,
                                 action_scheme='dynamic',
                                 reward_scheme='simple',
                                 feed=feed)
        assert env.action_scheme.pairs == [USD/BTC, USD/ETH]


def test_init(exchange, portfolio, feed):

    # Test 1
    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme="managed-risk",
        reward_scheme="simple",
        feed=feed
    )
    assert env

    # Test 2
    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme="dynamic",
        reward_scheme="simple",
        feed=feed
    )

    assert env

    # Test 3
    action_scheme = ManagedRiskOrders() + DynamicOrders()
    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme="simple",
        feed=feed,
        window_size=20
    )

    obs = env.reset()

    assert env
    assert obs.shape == (20, 11)

