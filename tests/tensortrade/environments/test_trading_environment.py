
import pandas as pd
import pytest

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges import Exchange
from tensortrade.instruments import USD, BTC, ETH, LTC
from tensortrade.rewards import SimpleProfit
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.actions import ManagedRiskOrders, DynamicOrders
from tensortrade.data import DataFeed, Array
from tensortrade.exchanges.services.execution.simulated import execute_order


@pytest.fixture
def portfolio():

    df1 = pd.read_csv("tests/data/input/coinbase_(BTC,ETH)USD_d.csv").tail(25)
    df1 = df1.rename({"Unnamed: 0": "date"}, axis=1)
    df1 = df1.set_index("date")

    df2 = pd.read_csv("tests/data/input/bitstamp_(BTC,ETH,LTC)USD_d.csv").tail(25)
    df2 = df2.rename({"Unnamed: 0": "date"}, axis=1)
    df2 = df2.set_index("date")

    ex1 = Exchange("coinbase", service=execute_order)(
        Array("USD-BTC", list(df1['BTC:close'])),
        Array("USD-ETH", list(df1['ETH:close']))
    )

    ex2 = Exchange("binance", service=execute_order)(
        Array("USD-BTC", list(df2['BTC:close'])),
        Array("USD-ETH", list(df2['ETH:close'])),
        Array("USD-LTC", list(df2['LTC:close']))
    )

    p = Portfolio(USD, [
        Wallet(ex1, 10000 * USD),
        Wallet(ex1, 10 * BTC),
        Wallet(ex1, 5 * ETH),
        Wallet(ex2, 1000 * USD),
        Wallet(ex2, 5 * BTC),
        Wallet(ex2, 20 * ETH),
        Wallet(ex2, 3 * LTC),
    ])
    return p


def test_init_multiple_exchanges(portfolio):

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=50
    )

    obs = env.reset()

    assert obs.shape == (50, 32)

    assert env.observation_space.shape == (50, 32)


def test_init_multiple_exchanges_with_external_feed(portfolio):
    pass


def test_runs_with_only_internal_data_feed(portfolio):

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=50
    )

    done = False
    obs = env.reset()
    while not done:

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    assert done