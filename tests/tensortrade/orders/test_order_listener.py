
import pytest
import unittest.mock as mock


from tensortrade.orders import OrderListener, Order
from tensortrade.instruments import USD, BTC
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.trades import TradeType, TradeSide


@pytest.fixture
def execute_listener():

    class ExecuteListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_execute(self, order: 'Order', exchange: 'Exchange'):
            self.listened = True

    return ExecuteListener()


@pytest.fixture
def fill_listener():

    class FillListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_fill(self, order: 'Order', exchange: 'Exchange', trade: 'Trade'):
            self.listened = True

    return FillListener()


@pytest.fixture
def complete_listener():

    class CompleteListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_complete(self, order: 'Order', exchange: 'Exchange'):
            self.listened = True

    return CompleteListener()


@pytest.fixture
def cancel_listener():

    class CancelListener(OrderListener):

        def __init__(self):
            self.listened = False

        def on_cancel(self, order: 'Order', exchange: 'Exchange'):
            self.listened = True

    return CancelListener()


@mock.patch('tensortrade.exchanges.Exchange')
def test_on_execute(mock_exchange_class,
                    execute_listener):
    exchange = mock_exchange_class.return_value
    exchange.id = "fake_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.attach(execute_listener)

    assert not execute_listener.listened
    order.execute(exchange)
    assert execute_listener.listened


@mock.patch('tensortrade.exchanges.Exchange')
def test_on_cancel(mock_exchange_class,
                   cancel_listener):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.attach(cancel_listener)

    assert not cancel_listener.listened
    order.cancel(exchange)
    assert cancel_listener.listened


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
def test_on_fill(mock_trade_class,
                 mock_exchange_class,
                 fill_listener):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.attach(fill_listener)

    order.execute(exchange)

    trade = mock_trade_class.return_value
    trade.size = 3997.00
    trade.commission = 3.00 * USD

    assert not fill_listener.listened
    order.fill(exchange, trade)
    assert fill_listener.listened


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
def test_on_complete(mock_trade_class,
                     mock_exchange_class,
                     complete_listener):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.attach(complete_listener)

    order.execute(exchange)

    trade = mock_trade_class.return_value
    trade.size = 5217.00
    trade.commission = 3.00 * USD

    order.fill(exchange, trade)

    assert not complete_listener.listened
    order.complete(exchange)
    assert complete_listener.listened
