
import unittest.mock as mock

from tensortrade.oms.orders.criteria import Criteria, Limit, Stop, Timed
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders import TradeSide


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_and(mock_order_class, mock_exchange_class):

    # Test initialization
    criteria = Limit(limit_price=7000.00) & Timed(duration=5)
    assert criteria
    assert isinstance(criteria, Criteria)

    order = mock_order_class.return_value
    order.pair = USD/BTC
    order.created_at = 0

    exchange = mock_exchange_class.return_value

    # Test (limit=True, timed=True) ===> True
    order.clock.step = 3

    order.side = TradeSide.BUY
    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert criteria(order, exchange)

    order.side = TradeSide.SELL
    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert criteria(order, exchange)

    # Test (limit=True, timed=False) ===> False
    order.clock.step = 8

    order.side = TradeSide.BUY
    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert not criteria(order, exchange)

    order.side = TradeSide.SELL
    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert not criteria(order, exchange)

    # Test (limit=False, timed=True) ===> False
    order.clock.step = 3

    order.side = TradeSide.BUY
    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert not criteria(order, exchange)

    order.side = TradeSide.SELL
    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert not criteria(order, exchange)

    # Test (limit=False, timed=False) ===> False
    order.clock.step = 8

    order.side = TradeSide.BUY
    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert not criteria(order, exchange)

    order.side = TradeSide.SELL
    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert not criteria(order, exchange)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_or(mock_order_class, mock_exchange_class):

    # Test Initialization
    criteria = Stop("down", 0.03) | Stop("up", 0.03)
    assert criteria
    assert isinstance(criteria, Criteria)

    order = mock_order_class.return_value
    order.pair = USD/BTC
    order.price = 7000.00

    exchange = mock_exchange_class.return_value

    # Test (stop=True, stop=False) ===> True
    # Greater than 3.00% below order price
    exchange.quote_price.return_value = 0.95 * order.price
    assert criteria(order, exchange)

    # Equal to 3.00% below order price
    exchange.quote_price.return_value = 0.969 * order.price
    assert criteria(order, exchange)

    # Test (stop=False, stop=False) ===> False
    # Less than 3.00% below order price
    exchange.quote_price.return_value = 0.98 * order.price
    assert not criteria(order, exchange)

    # Less than 3.00% above order price
    exchange.quote_price.return_value = 1.02 * order.price
    assert not criteria(order, exchange)

    # Test (stop=False, stop=True) ===> True
    # Equal to 3.00% above order price
    exchange.quote_price.return_value = 1.031 * order.price
    assert criteria(order, exchange)

    # Greater than 3.00% above order price
    exchange.quote_price.return_value = 1.05 * order.price
    assert criteria(order, exchange)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_xor(mock_order_class, mock_exchange_class):

    # Test Initialization
    criteria = Stop("down", 0.03) ^ Stop("up", 0.03)
    assert criteria
    assert isinstance(criteria, Criteria)

    order = mock_order_class.return_value
    order.pair = USD/BTC
    order.price = 7000.00

    exchange = mock_exchange_class.return_value

    # Test (stop=True, stop=False) ===> True
    # Greater than 3.00% below order price
    exchange.quote_price.return_value = 0.95 * order.price
    assert criteria(order, exchange)

    # Equal to 3.00% below order price
    exchange.quote_price.return_value = 0.969 * order.price
    assert criteria(order, exchange)

    # Test (stop=False, stop=False) ===> False
    # Less than 3.00% below order price
    exchange.quote_price.return_value = 0.98 * order.price
    assert not criteria(order, exchange)

    # Less than 3.00% above order price
    exchange.quote_price.return_value = 1.02 * order.price
    assert not criteria(order, exchange)

    # Test (stop=False, stop=True) ===> True
    # Equal to 3.00% above order price
    exchange.quote_price.return_value = 1.031 * order.price
    assert criteria(order, exchange)

    # Greater than 3.00% above order price
    exchange.quote_price.return_value = 1.05 * order.price
    assert criteria(order, exchange)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_invert(mock_order_class, mock_exchange_class):

    # Test Initialization
    criteria = ~Timed(5)
    assert criteria
    assert isinstance(criteria, Criteria)

    order = mock_order_class.return_value
    order.created_at = 0

    exchange = mock_exchange_class.return_value

    order.clock.step = 3
    assert not criteria(order, exchange)

    order.clock.step = 5
    assert not criteria(order, exchange)

    order.clock.step = 7
    assert criteria(order, exchange)
