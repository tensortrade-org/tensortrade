

import operator

from tensortrade.wallets import Wallet, Portfolio
from tensortrade.instruments import Quantity, USD, BTC, ETH, LTC


from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.exchanges import Exchange
from tensortrade.data.internal import create_internal_feed
from tensortrade.data import DataFeed, Stream


def test_create_internal_data_feed():

    ex1 = Exchange("coinbase", service=execute_order)(
        Stream([7000, 7500, 8300]).rename("USD-BTC"),
        Stream([200, 212, 400]).rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream([7005, 7600, 8200]).rename("USD-BTC"),
        Stream([201, 208, 402]).rename("USD-ETH"),
        Stream([56, 52, 60]).rename("USD-LTC")
    )

    portfolio = Portfolio(USD, [
        Wallet(ex1, 10000 * USD),
        Wallet(ex1, 10 * BTC),
        Wallet(ex1, 5 * ETH),
        Wallet(ex2, 1000 * USD),
        Wallet(ex2, 5 * BTC),
        Wallet(ex2, 20 * ETH),
        Wallet(ex2, 3 * LTC),
    ])

    feed = create_internal_feed(portfolio)

    data = {
        "coinbase:/USD-BTC": 7000,
        "coinbase:/USD-ETH": 200,
        "coinbase:/USD:/free": 10000,
        "coinbase:/USD:/locked": 0,
        "coinbase:/USD:/total": 10000,
        "coinbase:/BTC:/free": 10,
        "coinbase:/BTC:/locked": 0,
        "coinbase:/BTC:/total": 10,
        "coinbase:/BTC:/worth": 7000 * 10,
        "coinbase:/ETH:/free": 5,
        "coinbase:/ETH:/locked": 0,
        "coinbase:/ETH:/total": 5,
        "coinbase:/ETH:/worth": 200 * 5,
        "binance:/USD-BTC": 7005,
        "binance:/USD-ETH": 201,
        "binance:/USD-LTC": 56,
        "binance:/USD:/free": 1000,
        "binance:/USD:/locked": 0,
        "binance:/USD:/total": 1000,
        "binance:/BTC:/free": 5,
        "binance:/BTC:/locked": 0,
        "binance:/BTC:/total": 5,
        "binance:/BTC:/worth": 7005 * 5,
        "binance:/ETH:/free": 20,
        "binance:/ETH:/locked": 0,
        "binance:/ETH:/total": 20,
        "binance:/ETH:/worth": 201 * 20,
        "binance:/LTC:/free": 3,
        "binance:/LTC:/locked": 0,
        "binance:/LTC:/total": 3,
        "binance:/LTC:/worth": 56 * 3,
    }

    coinbase_net_worth = 10000 + (10 * 7000) + (5 * 200)
    binance_net_worth = 1000 + (5 * 7005) + (20 * 201) + (3 * 56)

    data['net_worth'] = sum(data[k] if k.endswith("worth") or k.endswith("USD:/total") else 0 for k in data.keys())

    assert feed.next() == data
