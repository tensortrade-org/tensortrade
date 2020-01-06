
import unittest.mock as mock
import re

from tensortrade.instruments import *
from tensortrade.orders import Order, OrderStatus, OrderSpec
from tensortrade.orders.criteria import Stop
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.trades import TradeSide, TradeType


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

    assert order.status == OrderStatus.OPEN
    order.fill(exchange, trade)
    assert order.status == OrderStatus.PARTIALLY_FILLED

    assert order.remaining_size == 1200.00

    listener.on_fill.assert_called_once_with(order, exchange, trade)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
@mock.patch('tensortrade.orders.OrderListener')
def test_complete_basic_order(mock_order_listener_class,
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
    listener.on_complete = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute(exchange)

    trade = mock_trade_class.return_value
    trade.size = 5217.00
    trade.commission = 3.00 * USD

    order.fill(exchange, trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete(exchange)
    assert order.status == OrderStatus.FILLED

    listener.on_complete.assert_called_once_with(order, exchange)
    assert not next_order


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
def test_complete_complex_order(mock_trade_class,
                                mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

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

    order.execute(exchange)

    # Execute fake trade
    price = 7010.00
    scale = order.price / price
    commission = 3.00 * USD

    base_size = scale * order.size - commission.size

    trade = mock_trade_class.return_value
    trade.size = base_size
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + trade.commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet -= Quantity(USD, size=base_size, path_id=order.path_id)
    quote_wallet += Quantity(BTC, size=quote_size, path_id=order.path_id)

    # Fill fake trade
    order.fill(exchange, trade)

    assert order.path_id in portfolio.get_wallet(exchange.id, USD).locked

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete(exchange)
    assert order.status == OrderStatus.FILLED

    assert next_order
    assert next_order.path_id == order.path_id
    assert next_order.size
    assert next_order.status == OrderStatus.PENDING
    assert next_order.side == TradeSide.SELL
    assert next_order.pair == USD/BTC


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.trades.Trade')
@mock.patch('tensortrade.orders.OrderListener')
def test_cancel(mock_order_listener_class,
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
    listener.on_cancel = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute(exchange)

    # Execute fake trade
    price = 7010.00
    scale = order.price / price
    commission = 3.00 * USD

    trade = mock_trade_class.return_value
    trade.size = scale * order.size - commission.size
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet -= Quantity(USD, size=base_size, path_id=order.path_id)
    quote_wallet += Quantity(BTC, size=quote_size, path_id=order.path_id)

    order.fill(exchange, trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert base_wallet.balance == 4800.00 * USD
    assert base_wallet.locked[order.path_id] == 7.42 * USD
    assert quote_wallet.balance == 0 * BTC
    assert quote_wallet.locked[order.path_id] == 0.73925519 * BTC
    order.cancel(exchange)

    listener.on_cancel.assert_called_once_with(order, exchange)
    assert base_wallet.balance == 4807.42 * USD
    assert order.path_id not in base_wallet.locked
    assert quote_wallet.balance == 0.73925519 * BTC
    assert order.path_id not in quote_wallet.locked


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


@mock.patch('tensortrade.wallets.Portfolio')
def test_str(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    pattern = re.compile("<[A-Z][a-zA-Z]*:\\s(\\w+=.*,\\s)*(\\w+=.*)>")

    string = str(order)
    assert string

    assert string == pattern.fullmatch(string).string


