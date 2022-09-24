
import re
import unittest.mock as mock

from decimal import Decimal

from tensortrade.oms.instruments import *
from tensortrade.oms.orders import Order, OrderStatus, OrderSpec, TradeSide, TradeType
from tensortrade.oms.orders.criteria import Stop
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.wallets import Wallet, MarginWallet, Portfolio


@mock.patch('tensortrade.exchanges.Exchange')
def test_init(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000 * USD,
                  price=7000,
                  portfolio=portfolio)

    assert order
    assert order.id
    assert order.path_id
    assert order.step == 0
    assert order.quantity.instrument == USD
    assert order.remaining == order.quantity
    assert isinstance(order.pair, TradingPair)
    assert order.pair.base == USD
    assert order.pair.quote == BTC


@mock.patch('tensortrade.exchanges.Exchange')
def test_properties(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.LIMIT,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    assert order
    assert order.step == 0
    assert order.base_instrument == USD
    assert order.quote_instrument == BTC
    assert order.size == 5000.00 * USD
    assert order.price == 7000.00
    assert order.trades == []
    assert order.is_buy
    assert not order.is_sell
    assert not order.is_market_order
    assert order.is_limit_order


@mock.patch('tensortrade.exchanges.Exchange')
def test_is_executable_on(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    # Market order
    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    exchange.quote_price = mock.Mock(return_value=Decimal(6800.00))
    assert order.is_executable

    exchange.quote_price = mock.Mock(return_value=Decimal(7200.00))
    assert order.is_executable

    # Limit order
    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.LIMIT,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    exchange.quote_price = mock.Mock(return_value=Decimal(6800.00))
    assert order.is_executable

    exchange.quote_price = mock.Mock(return_value=Decimal(7200.00))
    assert order.is_executable

    # Stop Order
    wallets = [Wallet(exchange, 0 * USD), Wallet(exchange, 2 * BTC)]
    portfolio = Portfolio(USD, wallets)
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.SELL,
                  trade_type=TradeType.LIMIT,
                  quantity=1 * BTC,
                  portfolio=portfolio,
                  price=Decimal(7000.00),
                  criteria=Stop("down", 0.03))

    exchange.quote_price = mock.Mock(return_value=Decimal(1 - 0.031) * order.price)
    assert order.is_executable

    exchange.quote_price = mock.Mock(return_value=Decimal(1 - 0.02) * order.price)
    assert not order.is_executable


@mock.patch("tensortrade.exchanges.Exchange")
def test_is_complete(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    # Market order
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    assert not order.is_complete

    order.remaining = 0 * USD
    assert order.is_complete


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.OrderSpec")
def test_add_order_spec(mock_order_spec_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    # Market order
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    order_spec = mock_order_spec_class.return_value

    assert len(order._specs) == 0
    order.add_order_spec(order_spec)
    assert len(order._specs) == 1

    assert order_spec in order._specs


@mock.patch("tensortrade.orders.OrderListener")
@mock.patch("tensortrade.exchanges.Exchange")
def test_attach(mock_exchange_class, mock_order_listener_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    listener = mock_order_listener_class.return_value

    assert len(order.listeners) == 0
    order.attach(listener)
    assert len(order.listeners) == 1
    assert listener in order.listeners


@mock.patch("tensortrade.orders.OrderListener")
@mock.patch("tensortrade.exchanges.Exchange")
def test_detach(mock_exchange_class, mock_order_listener_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    listener = mock_order_listener_class.return_value
    order.attach(listener)
    assert len(order.listeners) == 1
    assert listener in order.listeners

    order.detach(listener)
    assert len(order.listeners) == 0
    assert listener not in order.listeners


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.OrderListener')
def test_execute(mock_order_listener_class,
                 mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    listener = mock_order_listener_class.return_value
    listener.on_execute = mock.Mock(return_value=None)
    order.attach(listener)

    assert order.status == OrderStatus.PENDING
    order.execute()
    assert order.status == OrderStatus.OPEN

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    wallet_btc = portfolio.get_wallet(exchange.id, BTC)

    assert wallet_usd.balance == 4800 * USD
    assert wallet_usd.locked_balance == 5200 * USD
    assert order.path_id in wallet_usd.locked.keys()
    assert wallet_btc.balance == 0 * BTC

    listener.on_execute.assert_called_once_with(order)


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
@mock.patch("tensortrade.orders.OrderListener")
def test_fill(mock_order_listener_class,
              mock_trade_class,
              mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    listener = mock_order_listener_class.return_value
    listener.on_fill = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute()

    trade = mock_trade_class.return_value
    trade.size = Decimal(3997.00)
    trade.quantity = 3997.00 * USD
    trade.commission = 3.00 * USD

    assert order.status == OrderStatus.OPEN
    order.fill(trade)
    assert order.status == OrderStatus.PARTIALLY_FILLED

    assert order.remaining == 1200.00

    listener.on_fill.assert_called_once_with(order, trade)


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
@mock.patch("tensortrade.orders.OrderListener")
def test_complete_basic_order(mock_order_listener_class,
                              mock_trade_class,
                              mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    listener = mock_order_listener_class.return_value
    listener.on_complete = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute()

    trade = mock_trade_class.return_value
    trade.size = Decimal(5197.00)
    trade.quantity = 5197.00 * USD
    trade.commission = 3.00 * USD

    order.fill(trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete()
    assert order.status == OrderStatus.FILLED

    listener.on_complete.assert_called_once_with(order)
    assert not next_order


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
def test_complete_complex_order(mock_trade_class,
                                mock_exchange_class):
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

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

    order.execute()

    # Execute fake trade
    price = Decimal(7010.00)
    scale = order.price / price
    commission = 3.00 * USD

    base_size = scale * order.size - commission.size

    trade = mock_trade_class.return_value
    trade.size = Decimal(base_size)
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

    # Fill fake trade
    order.fill(trade)

    assert order.path_id in portfolio.get_wallet(exchange.id, USD).locked

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete()
    assert order.status == OrderStatus.FILLED

    assert next_order
    assert next_order.path_id == order.path_id
    assert next_order.size
    assert next_order.status == OrderStatus.PENDING
    assert next_order.side == TradeSide.SELL
    assert next_order.exchange_pair == ExchangePair(exchange, USD / BTC)


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
@mock.patch("tensortrade.orders.OrderListener")
def test_cancel(mock_order_listener_class,
                mock_trade_class,
                mock_exchange_class):

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

    listener = mock_order_listener_class.return_value
    listener.on_cancel = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute()

    # Execute fake trade
    price = Decimal(7010.00)
    scale = order.price / price
    commission = 3.00 * USD

    trade = mock_trade_class.return_value
    trade.size = Decimal(scale * order.size - commission.size)
    trade.quantity = trade.size * USD
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet.withdraw(
        quantity=Quantity(USD, size=base_size, path_id=order.path_id),
        reason="test"
    )
    quote_wallet.deposit(
        quantity=Quantity(BTC, size=quote_size, path_id=order.path_id),
        reason="test"
    )

    order.fill(trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert base_wallet.balance == 4800.00 * USD
    assert float(round(base_wallet.locked[order.path_id].size, 2)) == 7.42
    assert quote_wallet.balance == 0 * BTC
    assert float(round(quote_wallet.locked[order.path_id].size, 8)) == 0.73925519
    order.cancel()

    listener.on_cancel.assert_called_once_with(order)
    assert float(round(base_wallet.balance.size, 2)) == 4807.42
    assert order.path_id not in base_wallet.locked
    assert float(round(quote_wallet.balance.size, 8)) == 0.73925519
    assert order.path_id not in quote_wallet.locked


@mock.patch("tensortrade.exchanges.Exchange")
def test_release(mock_exchange_class):

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

    order.execute()

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    assert wallet_usd.balance == 4800 * USD
    assert wallet_usd.locked_balance == 5200 * USD
    assert order.path_id in wallet_usd.locked.keys()

    order.release()

    assert wallet_usd.balance == 10000 * USD
    assert wallet_usd.locked_balance == 0 * USD
    assert order.path_id not in wallet_usd.locked.keys()


@mock.patch("tensortrade.exchanges.Exchange")
def test_to_json(mock_exchange_class):

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

    d = {
        "id": str(order.id),
        "step": int(order.step),
        "exchange_pair": str(order.exchange_pair),
        "status": str(order.status),
        "type": str(order.type),
        "side": str(order.side),
        "base_symbol": str(order.pair.base.symbol),
        "quote_symbol": str(order.pair.quote.symbol),
        "quantity": str(order.quantity),
        "size": float(order.size),
        "remaining": str(order.remaining),
        "price": float(order.price),
        "criteria": str(order.criteria),
        "path_id": str(order.path_id),
        "created_at": str(order.created_at)
    }

    assert order.to_json() == d


@mock.patch("tensortrade.orders.OrderSpec")
@mock.patch("tensortrade.exchanges.Exchange")
def test_iadd(mock_exchange_class, mock_order_spec_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    # Market order
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=Decimal(7000.00))

    order_spec = mock_order_spec_class.return_value

    assert len(order._specs) == 0
    order.add_order_spec(order_spec)
    assert len(order._specs) == 1

    assert order_spec in order._specs


@mock.patch("tensortrade.exchanges.Exchange")
def test_str(mock_exchange_class):

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

    pattern = re.compile("<[A-Z][a-zA-Z]*:\\s(\\w+=.*,\\s)*(\\w+=.*)>")

    string = str(order)
    assert string

    assert string == pattern.fullmatch(string).string


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Trade")
@mock.patch("tensortrade.orders.OrderListener")
def test_complete_shorting_order(mock_order_listener_class,
                              mock_trade_class,
                              mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    wallets = [Wallet(exchange, 10000 * USD), MarginWallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))
    order = Order(step=0,
                  exchange_pair=ExchangePair(exchange, USD / BTC),
                  side=TradeSide.SELL,
                  trade_type=TradeType.MARKET,
                  quantity=0.5 * BTC,
                  portfolio=portfolio,
                  price=Decimal(7000.00))
    assert order.quantity.as_float() == 0.5
    assert order.remaining.as_float() == 0.5

    listener = mock_order_listener_class.return_value
    listener.on_complete = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute()

    # # Execute fake trade
    price = order.price
    scale = order.price / price
    commission = 0.005 * BTC

    trade = mock_trade_class.return_value
    trade.size = Decimal(scale * order.size - commission.size)
    trade.quantity = trade.size * BTC
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + commission.size
    quote_size = (order.price / trade.price) * (trade.size * trade.price)

    quote_wallet.withdraw(
        quantity=Quantity(BTC, size=float(trade.size)+trade.commission.as_float(), path_id=order.path_id),
        reason="test"
    )
    base_wallet.deposit(
        quantity=Quantity(USD, size=quote_size, path_id=order.path_id),
        reason="test"
    )

    order.fill(trade)

    # trade = mock_trade_class.return_value
    # trade.size = Decimal(-0.5)
    # trade.quantity = NegativeQuantity(BTC, -0.5)
    # trade.commission = NegativeQuantity(BTC, -0.005)

    # order.fill(trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete()
    assert order.status == OrderStatus.FILLED

    assert base_wallet.total_balance.as_float() == 13465
    assert quote_wallet.total_balance.as_float() == -0.5

    listener.on_complete.assert_called_once_with(order)
    assert not next_order