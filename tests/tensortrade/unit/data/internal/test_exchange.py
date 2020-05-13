

import pytest

from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.exchanges import Exchange
from tensortrade.data import DataFeed, Stream


def test_price_ds():

    btc_price = Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC")
    eth_price = Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")

    feed = DataFeed([btc_price, eth_price])

    assert feed.next() == {"USD-BTC": 7000, "USD-ETH": 200}


def test_exchange_feed():

    btc_price = Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC")
    eth_price = Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")

    exchange = Exchange("coinbase", service=execute_order)(
        btc_price,
        eth_price
    )

    feed = DataFeed(exchange.streams())

    assert feed.next() == {"coinbase:/USD-BTC": 7000, "coinbase:/USD-ETH": 200}
