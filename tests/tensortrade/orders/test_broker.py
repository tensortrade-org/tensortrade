
import pytest
import unittest.mock as mock

from tensortrade.orders import Broker, OrderStatus, Order, OrderSpec
from tensortrade.orders.criteria import Stop
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.trades import TradeSide, TradeType
from tensortrade.instruments import USD, BTC, Quantity


@mock.patch('tensortrade.exchanges.Exchange')
def test_init(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker(exchange)
    assert broker
    assert broker.exchanges == [exchange]
    assert broker.unexecuted == []
    assert broker.executed == {}
    assert broker.trades == {}

    exchanges = [
        mock_exchange_class.return_value,
        mock_exchange_class.return_value,
        mock_exchange_class.return_value
    ]
    broker = Broker(exchanges)
    assert broker
    assert broker.exchanges == exchanges
    assert broker.unexecuted == []
    assert broker.executed == {}
    assert broker.trades == {}


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_submit(mock_order_class,
                mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker(exchange)

    order = mock_order_class.return_value

    assert broker.unexecuted == []
    broker.submit(order)
    assert order in broker.unexecuted


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_cancel_unexecuted_order(mock_order_class,
                                 mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker(exchange)

    order = mock_order_class.return_value
    order.cancel = mock.Mock(return_value=None)
    order.status = OrderStatus.PENDING

    broker.submit(order)
    assert order in broker.unexecuted

    broker.cancel(order, exchange)
    assert order not in broker.unexecuted
    order.cancel.assert_called_once_with(exchange)


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_cancel_executed_order(mock_order_class,
                               mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker(exchange)

    order = mock_order_class.return_value
    order.cancel = mock.Mock(return_value=None)

    broker.submit(order)
    assert order in broker.unexecuted

    order.status = OrderStatus.OPEN
    with pytest.raises(Warning):
        broker.cancel(order, exchange)

    order.status = OrderStatus.PARTIALLY_FILLED
    with pytest.raises(Warning):
        broker.cancel(order, exchange)

    order.status = OrderStatus.FILLED
    with pytest.raises(Warning):
        broker.cancel(order, exchange)

    order.status = OrderStatus.CANCELLED
    with pytest.raises(Warning):
        broker.cancel(order, exchange)


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_update_on_single_exchange_with_single_order(mock_order_class,
                                                     mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker(exchange)

    order = mock_order_class.return_value
    order.id = "fake_id"
    order.is_executable_on = mock.Mock(side_effect=[False, True])
    order.attach = mock.Mock(return_value=None)

    broker.submit(order)

    # Test order does not execute on first update
    broker.update()
    assert order in broker.unexecuted
    assert order.id not in broker.executed

    # Test order does execute on second update
    broker.update()
    assert order not in broker.unexecuted
    assert order.id in broker.executed
    order.attach.assert_called_once_with(broker)


@mock.patch('tensortrade.exchanges.Exchange')
def test_update_on_single_exchange_with_multiple_orders(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    broker = Broker(exchange)

    # Submit order 1
    o1 = Order(side=TradeSide.BUY,
               trade_type=TradeType.MARKET,
               pair=USD / BTC,
               quantity=5200.00 * USD,
               portfolio=portfolio,
               price=7000.00)
    o1.is_executable_on = mock.MagicMock(side_effect=[False, True])
    broker.submit(o1)

    # Submit order 2
    o2 = Order(side=TradeSide.BUY,
               trade_type=TradeType.MARKET,
               pair=USD / BTC,
               quantity=230.00 * USD,
               portfolio=portfolio,
               price=7300.00)
    o2.is_executable_on = mock.MagicMock(side_effect=[True, False])
    broker.submit(o2)

    # No updates have been made yet
    assert o1 in broker.unexecuted and o1 not in broker.executed
    assert o2 in broker.unexecuted and o2 not in broker.executed

    # First update
    broker.update()
    assert o1 in broker.unexecuted and o1.id not in broker.executed
    assert o2 not in broker.unexecuted and o2.id in broker.executed

    # Second update
    broker.update()
    assert o1 not in broker.unexecuted and o1.id in broker.executed
    assert o2 not in broker.unexecuted and o2.id in broker.executed


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
def test_on_fill(mock_trade_class,
                 mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    broker = Broker(exchange)

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.attach(broker)

    order.execute(exchange)

    broker._executed[order.id] = order

    trade = mock_trade_class.return_value
    trade.size = 5197.00
    trade.commission = 3.00 * USD
    trade.order_id = order.id

    assert order.status == OrderStatus.OPEN
    order.fill(exchange, trade)
    assert order.status == OrderStatus.FILLED

    assert order.remaining_size == 0

    assert trade in broker.trades[order.id]


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
def test_on_fill_with_complex_order(mock_trade_class,
                                    mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    broker = Broker(exchange)

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    side = TradeSide.BUY

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    risk_criteria = Stop("down", 0.03) ^ Stop("up", 0.02)

    risk_management = OrderSpec(side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
                                trade_type=TradeType.MARKET,
                                pair=USD / BTC,
                                criteria=risk_criteria)

    order += risk_management

    order.attach(broker)
    order.execute(exchange)

    broker._executed[order.id] = order

    # Execute fake trade
    price = 7000.00
    scale = order.price / price
    commission = 3.00 * USD

    base_size = scale * order.size - commission.size

    trade = mock_trade_class.return_value
    trade.order_id = order.id
    trade.size = base_size
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + trade.commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet -= Quantity(USD, size=base_size, path_id=order.path_id)
    quote_wallet += Quantity(BTC, size=quote_size, path_id=order.path_id)

    assert trade.order_id in broker.executed.keys()
    assert trade not in broker.trades
    assert broker.unexecuted == []

    order.fill(exchange, trade)

    assert order.remaining_size == 0
    assert trade in broker.trades[order.id]
    assert broker.unexecuted != []


@mock.patch('tensortrade.exchanges.Exchange')
def test_reset(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    broker = Broker(exchange)

    broker._unexecuted = [78, 98, 100]
    broker._executed = {'a': 1, 'b': 2}
    broker._trades = {'a': 2, 'b': 3}

    broker.reset()

    assert broker.unexecuted == []
    assert broker.executed == {}
    assert broker.trades == {}
