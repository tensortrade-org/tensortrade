
import pytest
import unittest.mock as mock

from decimal import Decimal

from tensortrade.oms.orders import OrderListener, Order, TradeType, TradeSide
from tensortrade.oms.instruments import USD, BTC, ExchangePair
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import ExchangeOptions


@pytest.fixture
def execute_listener():

    class ExecuteListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_execute(self, order: 'Order'):
            self.listened = True

    return ExecuteListener()


@pytest.fixture
def fill_listener():

    class FillListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_fill(self, order: 'Order', trade: 'Trade'):
            self.listened = True

    return FillListener()


@pytest.fixture
def complete_listener():

    class CompleteListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_complete(self, order: 'Order'):
            self.listened = True

    return CompleteListener()


@pytest.fixture
def cancel_listener():

    class CancelListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_cancel(self, order: 'Order'):
            self.listened = True

    return CancelListener()


@mock.patch("tensortrade.exchanges.Exchange")
def test_on_execute(mock_exchange_class, execute_listener):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    order.attach(execute_listener)

    assert not execute_listener.listened
    order.execute()
    assert execute_listener.listened


@mock.patch("tensortrade.exchanges.Exchange")
def test_on_cancel(mock_exchange_class, cancel_listener):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    order.attach(cancel_listener)

    assert not cancel_listener.listened
    order.cancel()
    assert cancel_listener.listened


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
def test_on_fill(mock_trade_class, mock_exchange_class, fill_listener):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    order.attach(fill_listener)

    order.execute()

    trade = mock_trade_class.return_value
    trade.size = Decimal(3997.00)
    trade.quantity = trade.size * USD
    trade.commission = 3.00 * USD

    assert not fill_listener.listened
    order.fill(trade)
    assert fill_listener.listened


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
def test_on_complete(mock_trade_class, mock_exchange_class, complete_listener):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    order.attach(complete_listener)

    order.execute()

    trade = mock_trade_class.return_value
    trade.size = Decimal(5197.00)
    trade.quantity = trade.size * USD
    trade.commission = 3.00 * USD

    order.fill(trade)

    assert not complete_listener.listened
    order.complete()
    assert complete_listener.listened
