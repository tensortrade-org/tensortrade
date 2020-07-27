
import unittest.mock as mock

from tensortrade.oms.orders.criteria import Timed


def test_init():
    criteria = Timed(duration=8)
    assert criteria
    assert criteria.duration == 8


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_call(mock_order_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value

    order = mock_order_class.return_value
    order.created_at = 5
    order.clock.step = 10

    criteria = Timed(duration=10)
    assert criteria(order, exchange)

    order.clock.step = 15
    assert criteria(order, exchange)

    order.clock.step = 16
    assert not criteria(order, exchange)


def test_str():
    criteria = Timed(duration=8)
    assert str(criteria) == "<Timed: duration=8>"
