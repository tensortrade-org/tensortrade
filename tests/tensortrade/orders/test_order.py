"""
Feature: Order management by a broker

  Background:
    Given a Broker
    Given a Trader

  Scenario: Trader created a market order
    Given Trader has created a market order
    When Trader submits a market order to Broker


"""

import pytest
import unittest.mock as mock

from tensortrade.instruments import *
from tensortrade.orders import Order, OrderStatus
from tensortrade.orders.criteria import Stop
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.trades import TradeSide, TradeType, Trade


@mock.patch('tensortrade.wallets.Portfolio')
def test_init(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD/BTC,
                  quantity=5000 * USD,
                  price=7000,
                  portfolio=portfolio)
    assert order
    assert order.id
    assert order.path_id
    assert order.quantity.instrument == USD
    assert order.filled_size == 0
    assert order.remaining_size == order.quantity
    assert isinstance(order.pair, TradingPair)
    assert order.pair.base == USD
    assert order.pair.quote == BTC


@mock.patch('tensortrade.wallets.Portfolio')
def test_properties(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.LIMIT,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    assert order
    assert order.base_instrument == USD
    assert order.quote_instrument == BTC
    assert order.size == 5000.00 * USD
    assert order.price == 7000.00
    assert order.trades == []
    assert order.is_buy
    assert not order.is_sell
    assert not order.is_market_order
    assert order.is_limit_order
    assert order.created_at == 0


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.wallets.Portfolio')
def test_is_executable_on(mock_portfolio_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value
    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert order.is_executable_on(exchange)

    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert order.is_executable_on(exchange)

    # Limit order
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.LIMIT,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert order.is_executable_on(exchange)

    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert order.is_executable_on(exchange)

    # Stop Order
    order = Order(side=TradeSide.SELL,
                  trade_type=TradeType.LIMIT,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00,
                  criteria=Stop("down", 0.03))

    exchange.quote_price = mock.Mock(return_value=(1 - 0.031)*order.price)
    assert order.is_executable_on(exchange)

    exchange.quote_price = mock.Mock(return_value=(1 - 0.02) * order.price)
    assert not order.is_executable_on(exchange)


@mock.patch("tensortrade.wallets.Portfolio")
def test_is_complete(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    assert not order.is_complete()

    order.remaining_size = 0
    assert order.is_complete()


@mock.patch('tensortrade.orders.OrderSpec')
@mock.patch('tensortrade.wallets.Portfolio')
def test_add_order_spec(mock_portfolio_class, mock_order_spec_class):

    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order_spec = mock_order_spec_class.return_value

    assert len(order._specs) == 0
    order.add_order_spec(order_spec)
    assert len(order._specs) == 1

    assert order_spec in order._specs


@mock.patch('tensortrade.orders.OrderListener')
@mock.patch('tensortrade.wallets.Portfolio')
def test_attach(mock_portfolio_class, mock_order_listener_class):

    portfolio = mock_portfolio_class.return_value
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value

    assert len(order._listeners) == 0
    order.attach(listener)
    assert len(order._listeners) == 1
    assert listener in order._listeners


@mock.patch('tensortrade.orders.OrderListener')
@mock.patch('tensortrade.wallets.Portfolio')
def test_detach(mock_portfolio_class, mock_order_listener_class):
    portfolio = mock_portfolio_class.return_value
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value
    order.attach(listener)
    assert len(order._listeners) == 1
    assert listener in order._listeners

    order.detach(listener)
    assert len(order._listeners) == 0
    assert listener not in order._listeners


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.OrderListener')
def test_execute(mock_order_listener_class,
                 mock_exchange_class):

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

    listener = mock_order_listener_class.return_value
    listener.on_execute = mock.Mock(return_value=None)
    order.attach(listener)

    assert order.status == OrderStatus.PENDING
    order.execute(exchange)
    assert order.status == OrderStatus.OPEN

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    wallet_btc = portfolio.get_wallet(exchange.id, BTC)
    assert wallet_usd.balance == 4800 * USD
    assert wallet_usd.locked_balance == 5200 * USD
    assert order.path_id in wallet_usd.locked.keys()
    assert wallet_btc.balance == 0 * BTC

    listener.on_execute.assert_called_once_with(order, exchange)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
@mock.patch('tensortrade.orders.OrderListener')
def test_fill(mock_order_listener_class,
              mock_trade_class,
              mock_exchange_class):

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

    listener = mock_order_listener_class.return_value
    listener.on_fill = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute(exchange)

    trade = mock_trade_class.return_value
    trade.size = 3997.00
    trade.commission = 3.00 * USD

    order.fill(exchange, trade)

    assert order.remaining_size == 1200.00

    listener.on_fill.assert_called_once_with(order, exchange, trade)


def test_complete():
    pytest.fail("Failed.")


def test_cancel():
    pytest.fail("Failed.")


@mock.patch('tensortrade.exchanges.Exchange')
def test_release(mock_exchange_class):

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

    order.execute(exchange)

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    assert wallet_usd.balance == 4800 * USD
    assert wallet_usd.locked_balance == 5200 * USD
    assert order.path_id in wallet_usd.locked.keys()

    order.release()

    assert wallet_usd.balance == 10000 * USD
    assert wallet_usd.locked_balance == 0 * USD
    assert order.path_id not in wallet_usd.locked.keys()


@mock.patch('tensortrade.wallets.Portfolio')
def test_to_dict(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    d = {
            "id": order.id,
            "status": order.status,
            "type": order.type,
            "side": order.side,
            "pair": order.pair,
            "quantity": order.quantity,
            "size": order.size,
            "price": order.price,
            "criteria": order.criteria,
            "path_id": order.path_id
    }

    assert order.to_dict() == d


@mock.patch('tensortrade.wallets.Portfolio')
def test_to_json(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    d = {
            "id": order.id,
            "status": order.status,
            "type": order.type,
            "side": order.side,
            "pair": order.pair,
            "quantity": order.quantity,
            "size": order.size,
            "price": order.price,
            "criteria": order.criteria,
            "path_id": order.path_id
    }

    d = {k: str(v) for k, v in d.items()}

    assert order.to_json() == d


@mock.patch('tensortrade.orders.OrderSpec')
@mock.patch('tensortrade.wallets.Portfolio')
def test_iadd(mock_portfolio_class, mock_order_spec_class):

    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order_spec = mock_order_spec_class.return_value

    assert len(order._specs) == 0
    order += order_spec
    assert len(order._specs) == 1

    assert order_spec in order._specs


def test_str():
    pytest.fail("Failed.")


