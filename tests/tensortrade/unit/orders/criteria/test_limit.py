
import unittest.mock as mock

from tensortrade.oms.orders.criteria import Limit
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders import TradeSide


def test_init():
    criteria = Limit(limit_price=7000.00)
    assert criteria.limit_price == 7000.00


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_call_with_buy_order(mock_order_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.quote_price = mock.Mock(return_value=7000.00)

    order = mock_order_class.return_value
    order.pair = USD/BTC
    order.side = TradeSide.BUY

    criteria = Limit(limit_price=6800.00)
    assert not criteria(order, exchange)

    criteria = Limit(limit_price=7000.00)
    assert criteria(order, exchange)

    criteria = Limit(limit_price=7200.00)
    assert criteria(order, exchange)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_call_with_sell_order(mock_order_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.quote_price.return_value = 7000.00

    order = mock_order_class.return_value
    order.pair = USD/BTC
    order.side = TradeSide.SELL

    criteria = Limit(limit_price=6800.00)
    assert criteria(order, exchange)

    criteria = Limit(limit_price=7000.00)
    assert criteria(order, exchange)

    criteria = Limit(limit_price=7200.00)
    assert not criteria(order, exchange)


def test_str():
    criteria = Limit(limit_price=7000.00)
    assert str(criteria) == "<Limit: price={0}>".format(7000.00)
