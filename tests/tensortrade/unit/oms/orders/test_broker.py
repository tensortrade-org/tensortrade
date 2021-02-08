
import pytest
import unittest.mock as mock

from decimal import Decimal

from tensortrade.oms.orders import Broker, OrderStatus, Order, OrderSpec, TradeSide, TradeType
from tensortrade.oms.orders.criteria import Stop
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC, Quantity, ExchangePair


@mock.patch('tensortrade.exchanges.Exchange')
def test_init(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker()
    broker.exchanges = [exchange]

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
    broker = Broker()
    broker.exchanges = exchanges
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
    broker = Broker()
    broker.exchanges = [exchange]

    order = mock_order_class.return_value

    assert broker.unexecuted == []
    broker.submit(order)
    assert order in broker.unexecuted


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_cancel_unexecuted_order(mock_order_class,
                                 mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker()
    broker.exchanges = [exchange]

    order = mock_order_class.return_value
    order.cancel = mock.Mock(return_value=None)
    order.status = OrderStatus.PENDING

    broker.submit(order)
    assert order in broker.unexecuted

    broker.cancel(order)
    assert order not in broker.unexecuted
    order.cancel.assert_called_once_with()


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_cancel_executed_order(mock_order_class,
                               mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options.max_trade_size = 1e6
    broker = Broker()
    broker.exchanges = [exchange]

    order = mock_order_class.return_value
    order.cancel = mock.Mock(return_value=None)

    broker.submit(order)
    assert order in broker.unexecuted

    order.status = OrderStatus.CANCELLED
    with pytest.raises(Warning):
        broker.cancel(order)


@mock.patch('tensortrade.orders.Order')
@mock.patch('tensortrade.exchanges.Exchange')
def test_update_on_single_exchange_with_single_order(mock_order_class,
                                                     mock_exchange_class):

    exchange = mock_exchange_class.return_value
    broker = Broker()
    broker.exchanges = [exchange]

    order = mock_order_class.return_value
    order.id = "fake_id"
    order.start = 0
    order.end = None
    order.is_executable_on = mock.Mock(side_effect=[False, True])
    order.attach = mock.Mock(return_value=None)

    broker.submit(order)

    # Test order does execute
    broker.update()
    assert order not in broker.unexecuted
    assert order.id in broker.executed
    order.attach.assert_called_once_with(broker)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Trade')
def test_on_fill(mock_trade_class,
                 mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options.max_trade_size = 1e6
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.quote_price = lambda pair: Decimal(7000.00)

    broker = Broker()
    broker.exchanges = [exchange]

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.attach(broker)

    order.execute()

    broker.executed[order.id] = order

    trade = mock_trade_class.return_value
    trade.quantity = 5197.00 * USD
    trade.commission = 3.00 * USD
    trade.order_id = order.id

    assert order.status == OrderStatus.OPEN
    order.fill(trade)
    assert order.status == OrderStatus.FILLED

    assert order.remaining == 0

    assert trade in broker.trades[order.id]


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Trade')
def test_on_fill_with_complex_order(mock_trade_class,
                                    mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options.max_trade_size = 1e6
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.quote_price = lambda pair: Decimal(7000.00)

    broker = Broker()
    broker.exchanges = [exchange]

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    side = TradeSide.BUY

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    risk_criteria = Stop("down", 0.03) ^ Stop("up", 0.02)

    risk_management = OrderSpec(side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
                                trade_type=TradeType.MARKET,
                                exchange_pair=ExchangePair(exchange, USD / BTC),
                                criteria=risk_criteria)

    order.add_order_spec(risk_management)

    order.attach(broker)
    order.execute()

    broker.executed[order.id] = order

    # Execute fake trade
    price = Decimal(7000.00)
    scale = order.price / price
    commission = 3.00 * USD

    base_size = scale * order.size - commission.size

    trade = mock_trade_class.return_value
    trade.order_id = order.id
    trade.size = base_size
    trade.quantity = base_size * USD
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + trade.commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet.withdraw(
        quantity=Quantity(USD, size=base_size, path_id=order.path_id),
        reason="test"
    )
    quote_wallet.deposit(
        quantity=Quantity(BTC, size=quote_size, path_id=order.path_id),
        reason="test"
    )

    assert trade.order_id in broker.executed.keys()
    assert trade not in broker.trades
    assert broker.unexecuted == []

    order.fill(trade)

    assert order.remaining == 0
    assert trade in broker.trades[order.id]
    assert broker.unexecuted != []


@mock.patch('tensortrade.exchanges.Exchange')
def test_reset(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    broker = Broker()
    broker.exchanges = [exchange]

    broker._unexecuted = [78, 98, 100]
    broker._executed = {'a': 1, 'b': 2}
    broker._trades = {'a': 2, 'b': 3}

    broker.reset()

    assert broker.unexecuted == []
    assert broker.executed == {}
    assert broker.trades == {}
