
import pytest
import unittest.mock as mock
import numpy as np
import pandas as pd

from tensortrade.data import DataFeed
from tensortrade.instruments import USD, BTC, ETH, LTC, Quantity
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.exchanges.simulated import SimulatedExchange


@pytest.fixture
def data_frame():
    index = pd.Index(
        ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
        name="date"
    )

    data_btc = np.array([
        [13863.13, 13889., 12952.5, 13480.01, 11484.01],
        [13480.01, 15275., 13005., 14781.51, 23957.87],
        [14781.51, 15400., 14628., 15098.14, 16584.63],
        [15098.14, 15400., 14230., 15144.99, 17980.39],
        [15144.99, 17178., 14824.05, 16960.01, 20781.65]
    ])
    btc_columns = ['BTC:open', 'BTC:high', 'BTC:low', 'BTC:close', 'BTC:volume']

    frame_btc = pd.DataFrame(data_btc, columns=btc_columns, index=index)

    data_eth = np.array([
        [1.198000e+01, 1.198000e+01, 1.025000e+01, 1.125000e+01, 1.374903e+04],
        [1.125000e+01, 1.244000e+01, 1.070000e+01, 1.193000e+01, 1.581034e+04],
        [1.193000e+01, 1.249000e+01, 1.165000e+01, 1.234000e+01, 3.481300e+03],
        [1.234000e+01, 1.288000e+01, 1.200000e+01, 1.241000e+01, 4.110590e+03],
        [1.241000e+01, 1.424000e+01, 1.237000e+01, 1.400000e+01, 5.654910e+03]
    ])
    eth_columns = ['ETH:open', 'ETH:high', 'ETH:low', 'ETH:close', 'ETH:volume']
    frame_eth = pd.DataFrame(data_eth, columns=eth_columns, index=index)

    data_frame = pd.concat([frame_btc, frame_eth], axis=1)

    return data_frame


@pytest.fixture
def exchange_ds(data_frame):

    ds = SimExchangeDS(
        name="coinbase",
        data_frame=data_frame,
        fetch=lambda x: {
            USD/BTC: x['BTC:close'],
            USD/ETH: x['ETH:close']
        }
    )
    return ds


@pytest.fixture
def exchange(exchange_ds):
    return SimulatedExchange(exchange_ds)


@pytest.fixture
def portfolio(exchange):
    portfolio = Portfolio(USD, [
        Wallet(exchange, 10000*USD),
        Wallet(exchange, 10 * BTC),
        Wallet(exchange, 200 * ETH)
    ])
    return portfolio


def tests_portfolio_data_source(exchange_ds, exchange, portfolio):

    source = PortfolioDS(
        portfolio,
        fetch=lambda x: {
            ("coinbase", USD / BTC): x['coinbase_BTC:close'],
            ("coinbase", USD / ETH): x['coinbase_ETH:close']
        }
    )
    source.use([exchange_ds])

    feed = DataFeed(inputs=[exchange_ds], outputs=[source])

    btc_price_0 = 13480.01
    eth_price_0 = 11.25

    net_worth = 10000.00 + (10.00000000 * btc_price_0) + (200.00000000 * eth_price_0)

    assert feed
    assert feed.next() == {'coinbase_USD': 10000.00, 'coinbase_USD_pending': 0.00,
                           'coinbase_BTC': 10.00000000, 'coinbase_BTC_pending': 0.00000000,
                           'coinbase_ETH': 200.00000000, 'coinbase_ETH_pending': 0.00000000,
                           'net_worth': net_worth}

    wallet_usd = portfolio.get_wallet(exchange.id, USD)

    wallet_usd -= 1000*USD
    wallet_usd += Quantity(USD, 1000, path_id="fake_path_id")

    btc_price_1 = 14781.51
    eth_price_1 = 11.93

    net_worth = 10000 + (10 * btc_price_1) + (200 * eth_price_1)

    assert feed.next() == {'coinbase_USD': 9000.00, 'coinbase_USD_pending': 1000.00,
                           'coinbase_BTC': 10.00000000, 'coinbase_BTC_pending': 0.00000000,
                           'coinbase_ETH': 200.00000000, 'coinbase_ETH_pending': 0.00000000,
                           'net_worth': net_worth}


def test_internal_data_feed(data_frame):

    exchange_ds = SimExchangeDS(
        name="coinbase",
        data_frame=data_frame,
        fetch=lambda x: {
                USD/BTC: x['BTC:close'],
                USD/ETH: x['ETH:close']
        }
    )

    exchange = SimulatedExchange(exchange_ds)

    portfolio = Portfolio(USD, [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 10 * BTC),
        Wallet(exchange, 200 * ETH)
    ])

    portfolio_ds = PortfolioDS(
        portfolio=portfolio,
        fetch=lambda x: {
            ("coinbase", USD/BTC): x['coinbase_BTC:close'],
            ("coinbase", USD/ETH): x['coinbase_ETH:close']
        }
    )
    portfolio_ds.use([exchange_ds])

    feed = DataFeed(
        inputs=[exchange_ds],
        outputs=[exchange_ds, portfolio_ds]
    )

    d = feed.next()

    d1 = {
        'coinbase_BTC:open': 13863.13,
        'coinbase_BTC:high': 13889.,
        'coinbase_BTC:low': 12952.5,
        'coinbase_BTC:close': 13480.01,
        'coinbase_BTC:volume': 11484.01,
        'coinbase_ETH:open': 11.98,
        'coinbase_ETH:high': 11.98,
        'coinbase_ETH:low': 10.25,
        'coinbase_ETH:close': 11.25,
        'coinbase_ETH:volume': 13749.03,
        'coinbase_USD': 10000,
        'coinbase_USD_pending': 0,
        'coinbase_BTC': 10,
        'coinbase_BTC_pending': 0,
        'coinbase_ETH': 200,
        'coinbase_ETH_pending': 0,
        'net_worth': 10000 + (10 * 13480.01) + (200 * 11.25)
    }

    print(d1)
    assert d1 == d

    wallet_usd = portfolio.get_wallet(exchange.id, USD)

    wallet_usd -= 1000 * USD
    wallet_usd += Quantity(USD, 1000, path_id="fake_path_id")

    d = feed.next()

    d2 = {
        'coinbase_BTC:open': 13480.01,
        'coinbase_BTC:high': 15275.,
        'coinbase_BTC:low': 13005.,
        'coinbase_BTC:close': 14781.51,
        'coinbase_BTC:volume': 23957.87,
        'coinbase_ETH:open': 11.25,
        'coinbase_ETH:high': 12.44,
        'coinbase_ETH:low': 10.70,
        'coinbase_ETH:close': 11.93,
        'coinbase_ETH:volume': 15810.34,
        'coinbase_USD': 9000,
        'coinbase_USD_pending': 1000,
        'coinbase_BTC': 10,
        'coinbase_BTC_pending': 0,
        'coinbase_ETH': 200,
        'coinbase_ETH_pending': 0,
        'net_worth': 10000 + (10 * 14781.51) + (200 * 11.93)
    }

    assert d == d2


def test_two_exchanges():

    df1 = pd.read_csv("tests/data/input/coinbase_(BTC,ETH)USD_d.csv").tail()
    df1 = df1.rename({"Unnamed: 0": "date"}, axis=1)
    df1 = df1.set_index("date")

    df2 = pd.read_csv("tests/data/input/bitstamp_(BTC,ETH,LTC)USD_d.csv").tail()
    df2 = df2.rename({"Unnamed: 0": "date"}, axis=1)
    df2 = df2.set_index("date")

    exchange_ds1 = SimExchangeDS(
        name="coinbase",
        data_frame=df1,
        fetch=lambda x: {
          USD / BTC: x['BTC:close'],
          USD / ETH: x['ETH:close']
    })

    exchange_ds2 = SimExchangeDS(
        name="bitstamp",
        data_frame=df2,
        fetch=lambda x: {
          USD / BTC: x['BTC:close'],
          USD / ETH: x['ETH:close'],
          USD / LTC: x['LTC:close']
        }
    )

    ex1 = SimulatedExchange(exchange_ds1)
    ex2 = SimulatedExchange(exchange_ds2)

    portfolio = Portfolio(USD, [
        Wallet(ex1, 1000 * USD),
        Wallet(ex1, 200 * BTC),
        Wallet(ex1, 200 * ETH),
        Wallet(ex2, 500 * USD),
        Wallet(ex2, 200 * BTC),
        Wallet(ex2, 500 * ETH),
        Wallet(ex2, 7000 * LTC)
    ])

    portfolio = PortfolioDS(
        portfolio=portfolio,
        fetch=lambda x: {
            ("coinbase", USD / BTC): x["coinbase_BTC:close"],
            ("coinbase", USD / ETH): x["coinbase_ETH:close"],
            ("bitstamp", USD / BTC): x["bitstamp_BTC:close"],
            ("bitstamp", USD / ETH): x["bitstamp_ETH:close"],
            ("bitstamp", USD / LTC): x["bitstamp_LTC:close"]
        }
    )
    portfolio.use([exchange_ds1, exchange_ds2])

    inputs = [exchange_ds1, exchange_ds2]
    outputs = [exchange_ds1, exchange_ds2, portfolio]
    feed = DataFeed(inputs=inputs, outputs=outputs)

    x = feed.next()

    coinbase_net_worth = 1000 + (200 * 8104.5) + (200 * 143.51)
    bitstamp_net_worth = 500 + (200 * 8105.01) + (500 * 143.51) + (7000 * 49.51)

    d = {
        'coinbase_BTC:open': 8180.81,
        'coinbase_BTC:high': 8195.81,
        'coinbase_BTC:low': 8041.95,
        'coinbase_BTC:close': 8104.5,
        'coinbase_BTC:volume': 6678.03,
        'coinbase_ETH:open': 146.6,
        'coinbase_ETH:high': 147.0,
        'coinbase_ETH:low': 142.1,
        'coinbase_ETH:close': 143.51,
        'coinbase_ETH:volume': 44072.66,
        'bitstamp_BTC:open': 8180.76,
        'bitstamp_BTC:high': 8196.81,
        'bitstamp_BTC:low': 8039.0,
        'bitstamp_BTC:close': 8105.01,
        'bitstamp_BTC:volume': 4011.44,
        'bitstamp_ETH:open': 146.28,
        'bitstamp_ETH:high': 146.86,
        'bitstamp_ETH:low': 142.14,
        'bitstamp_ETH:close': 143.51,
        'bitstamp_ETH:volume': 14097.82,
        'bitstamp_LTC:open': 51.33,
        'bitstamp_LTC:high': 51.47,
        'bitstamp_LTC:low': 48.88,
        'bitstamp_LTC:close': 49.51,
        'bitstamp_LTC:volume': 35569.13,
        'coinbase_USD': 1000,
        'coinbase_USD_pending': 0,
        'coinbase_BTC': 200,
        'coinbase_BTC_pending': 0,
        'coinbase_ETH': 200,
        'coinbase_ETH_pending': 0,
        'bitstamp_BTC': 200,
        'bitstamp_BTC_pending': 0,
        'bitstamp_USD': 500,
        'bitstamp_USD_pending': 0,
        'bitstamp_ETH': 500,
        'bitstamp_ETH_pending': 0,
        'bitstamp_LTC': 7000,
        'bitstamp_LTC_pending': 0,
        'net_worth': coinbase_net_worth + bitstamp_net_worth
    }

    assert x == d
