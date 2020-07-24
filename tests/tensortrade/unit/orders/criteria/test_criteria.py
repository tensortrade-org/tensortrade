
import unittest.mock as mock

from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.orders.criteria import Criteria
from tensortrade.oms.instruments import USD, BTC


class ConcreteCriteria(Criteria):

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        return True


def test_init():
    criteria = ConcreteCriteria()
    assert criteria


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_default_checks_on_call(mock_order_class, mock_exchange_class):

    order = mock_order_class.return_value
    order.pair = USD/BTC

    exchange = mock_exchange_class.return_value
    exchange.is_pair_tradable = lambda pair: str(pair) in ["USD/BTC", "USD/ETH", "USD/XRP"]

    criteria = ConcreteCriteria()

    # Pair being traded is supported by exchange
    exchange.is_pair_tradable = lambda pair: str(pair) in ["USD/BTC", "USD/ETH", "USD/XRP"]
    assert criteria(order, exchange)

    # Pair being traded is not supported by exchange
    exchange.is_pair_tradable = lambda pair: str(pair) in ["USD/ETH", "USD/XRP"]
    assert not criteria(order, exchange)






