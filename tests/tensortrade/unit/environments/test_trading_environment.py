
import pandas as pd
import pytest
import ta

from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges import Exchange
from tensortrade.instruments import USD, BTC, ETH, LTC
from tensortrade.rewards import SimpleProfit
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.actions import ManagedRiskOrders
from tensortrade.data import DataFeed, Stream, Module
from tensortrade.exchanges.services.execution.simulated import execute_order


@pytest.fixture
def portfolio():

    df1 = pd.read_csv("tests/data/input/coinbase_(BTC,ETH)USD_d.csv").tail(100)
    df1 = df1.rename({"Unnamed: 0": "date"}, axis=1)
    df1 = df1.set_index("date")

    df2 = pd.read_csv("tests/data/input/bitstamp_(BTC,ETH,LTC)USD_d.csv").tail(100)
    df2 = df2.rename({"Unnamed: 0": "date"}, axis=1)
    df2 = df2.set_index("date")

    ex1 = Exchange("coinbase", service=execute_order)(
        Stream(list(df1['BTC:close']), "USD-BTC"),
        Stream(list(df1['ETH:close']), "USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream(list(df2['BTC:close']), "USD-BTC"),
        Stream(list(df2['ETH:close']), "USD-ETH"),
        Stream(list(df2['LTC:close']), "USD-LTC")
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


def test_runs_with_external_feed_only(portfolio):

    df = pd.read_csv("tests/data/input/coinbase_(BTC,ETH)USD_d.csv").tail(100)
    df = df.rename({"Unnamed: 0": "date"}, axis=1)
    df = df.set_index("date")

    coinbase_btc = df.loc[:, [name.startswith("BTC") for name in df.columns]]
    coinbase_eth = df.loc[:, [name.startswith("ETH") for name in df.columns]]

    ta.add_all_ta_features(
        coinbase_btc,
        colprefix="BTC:",
        **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )
    ta.add_all_ta_features(
        coinbase_eth,
        colprefix="ETH:",
        **{k: "ETH:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )

    nodes = []
    with Module("coinbase") as coinbase:
        for name in coinbase_btc.columns:
            nodes += [Stream(list(coinbase_btc[name]), name)]
        for name in coinbase_eth.columns:
            nodes += [Stream(list(coinbase_eth[name]), name)]

    feed = DataFeed([coinbase])

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = TradingEnvironment(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=50,
        use_internal=False,
        enable_logger=False
    )

    done = False
    obs = env.reset()
    while not done:

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    assert obs.shape[0] == 50
