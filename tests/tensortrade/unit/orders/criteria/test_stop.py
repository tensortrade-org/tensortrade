
import unittest.mock as mock

from tensortrade.oms.orders.criteria import Stop, StopDirection
from tensortrade.oms.instruments import USD, BTC


def test_init():

    criteria = Stop(direction=StopDirection.UP, percent=0.3)
    assert criteria.direction == StopDirection.UP
    assert criteria.percent == 0.3


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_call_with_direction_down(mock_order_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value

    order = mock_order_class.return_value
    order.pair = USD/BTC
    order.price = 7000.00

    criteria = Stop(direction=StopDirection.DOWN, percent=0.03)

    # Greater than 3.00% below order price
    exchange.quote_price.return_value = 0.95 * order.price
    assert criteria(order, exchange)

    # Equal to 3.00% below order price
    exchange.quote_price.return_value = 0.969 * order.price
    assert criteria(order, exchange)

    # Less than 3.00% below order price
    exchange.quote_price.return_value = 0.98 * order.price
    assert not criteria(order, exchange)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_call_with_direction_up(mock_order_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value

    order = mock_order_class.return_value
    order.pair = USD / BTC
    order.price = 7000.00

    criteria = Stop(direction=StopDirection.UP, percent=0.03)

    # Less than 3.00% above order price
    exchange.quote_price.return_value = 1.02 * order.price
    assert not criteria(order, exchange)

    # Equal to 3.00% above order price
    exchange.quote_price.return_value = 1.031 * order.price
    assert criteria(order, exchange)

    # Greater than 3.00% above order price
    exchange.quote_price.return_value = 1.05 * order.price
    assert criteria(order, exchange)


def test_str():
    criteria = Stop(direction=StopDirection.UP, percent=0.3)
    assert str(criteria) == "<Stop: direction=up, percent=0.3>"
