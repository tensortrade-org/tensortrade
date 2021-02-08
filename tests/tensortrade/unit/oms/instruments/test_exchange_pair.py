import unittest.mock as mock

from tensortrade.oms.instruments import ExchangePair, BTC, USD


@mock.patch('tensortrade.exchanges.Exchange')
def test_valid_init(mock_exchange):

    exchange = mock_exchange.return_value
    exchange.name = "bitfinex"

    exchange_pair = ExchangePair(exchange, USD/BTC)
    assert exchange_pair
    assert exchange_pair.pair.base == USD
    assert exchange_pair.pair.quote == BTC


@mock.patch('tensortrade.exchanges.Exchange')
def test_str(mock_exchange):
    exchange = mock_exchange.return_value
    exchange.name = "bitfinex"

    exchange_pair = ExchangePair(exchange, USD/BTC)

    assert str(exchange_pair) == "bitfinex:USD/BTC"
