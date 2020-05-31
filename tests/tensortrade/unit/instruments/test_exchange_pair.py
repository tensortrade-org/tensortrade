

import pytest
import unittest.mock as mock

from tensortrade.instruments import ExchangePair, BTC, USD
from tensortrade.base.exceptions import InvalidTradingPair


@mock.patch('tensortrade.exchanges.Exchange')
def test_valid_init(mock_exchange):

    exchange = mock_exchange.return_value
    exchange.name = "coinbase"

    exchange_pair = ExchangePair(exchange, USD/BTC)
    assert exchange_pair
    assert exchange_pair.pair.base == USD
    assert exchange_pair.pair.quote == BTC


@mock.patch('tensortrade.exchanges.Exchange')
def test_str(mock_exchange):
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"

    exchange_pair = ExchangePair(exchange, USD/BTC)

    assert str(exchange_pair) == "coinbase:USD/BTC"
