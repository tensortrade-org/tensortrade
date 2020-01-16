
import pytest
import unittest.mock as mock
import numpy as np
import pandas as pd

from tensortrade.data import PortfolioDataSource, ExchangeDataSource, DataFeed
from tensortrade.instruments import USD, BTC, ETH, Quantity
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

    ds = ExchangeDataSource(data_frame=data_frame,
                            fetch=lambda x: {
                                USD/BTC: x['BTC:close'],
                                USD/ETH: x['ETH:close']
                            })
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

    source = PortfolioDataSource(portfolio,
                                 fetch=lambda x: {
                                     USD/BTC: x['BTC:close'],
                                     USD/ETH: x['ETH:close']
                                 })
    source.use([exchange_ds])

    feed = DataFeed(inputs=[exchange_ds], outputs=[source])

    btc_price_0 = 13480.01
    eth_price_0 = 11.25

    net_worth = 10000.00 + (10.00000000 * btc_price_0) + (200.00000000 * eth_price_0)

    assert feed
    assert feed.next() == {'USD': 10000.00, 'USD_pending': 0.00,
                           'BTC': 10.00000000, 'BTC_pending': 0.00000000,
                           'ETH': 200.00000000, 'ETH_pending': 0.00000000,
                           'net_worth': net_worth}

    wallet_usd = portfolio.get_wallet(exchange.id, USD)

    wallet_usd -= 1000*USD
    wallet_usd += Quantity(USD, 1000, path_id="fake_path_id")

    btc_price_1 = 14781.51
    eth_price_1 = 11.93

    net_worth = 10000 + (10 * btc_price_1) + (200 * eth_price_1)

    assert feed.next() == {'USD': 9000.00, 'USD_pending': 1000.00,
                           'BTC': 10.00000000, 'BTC_pending': 0.00000000,
                           'ETH': 200.00000000, 'ETH_pending': 0.00000000,
                           'net_worth': net_worth}


def test_internal_data_feed(data_frame):

    exchange_ds = ExchangeDataSource(data_frame=data_frame,
                                     fetch=lambda x: {
                                        USD/BTC: x['BTC:close'],
                                        USD/ETH: x['ETH:close']
                                     })

    exchange = SimulatedExchange(exchange_ds)

    portfolio = Portfolio(USD, [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 10 * BTC),
        Wallet(exchange, 200 * ETH)
    ])

    portfolio_ds = PortfolioDataSource(portfolio=portfolio,
                                       fetch=lambda x: {
                                           USD/BTC: x['BTC:close'],
                                           USD/ETH: x['ETH:close']
                                       })
    portfolio_ds.use([exchange_ds])

    feed = DataFeed(
        inputs=[exchange_ds],
        outputs=[exchange_ds, portfolio_ds]
    )

    d = feed.next()

    d1 = {
        'BTC:open': 13863.13,
        'BTC:high': 13889.,
        'BTC:low': 12952.5,
        'BTC:close': 13480.01,
        'BTC:volume': 11484.01,
        'ETH:open': 11.98,
        'ETH:high': 11.98,
        'ETH:low': 10.25,
        'ETH:close': 11.25,
        'ETH:volume': 13749.03,
        'USD': 10000,
        'USD_pending': 0,
        'BTC': 10,
        'BTC_pending': 0,
        'ETH': 200,
        'ETH_pending': 0,
        'net_worth': 10000 + (10 * 13480.01) + (200 * 11.25)
    }
    assert d1 == d

    wallet_usd = portfolio.get_wallet(exchange.id, USD)

    wallet_usd -= 1000 * USD
    wallet_usd += Quantity(USD, 1000, path_id="fake_path_id")

    d = feed.next()
