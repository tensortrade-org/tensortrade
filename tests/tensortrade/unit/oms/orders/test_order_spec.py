
import re
import unittest.mock as mock

from decimal import Decimal

from tensortrade.oms.orders import OrderSpec, TradeSide, TradeType
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.instruments import USD, BTC, ExchangePair
from tensortrade.oms.wallets import Portfolio, Wallet


@mock.patch("tensortrade.exchanges.Exchange")
def test_init(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    side = TradeSide.BUY
    trade_type = TradeType.MARKET

    # Create order specification without criteria
    order_spec = OrderSpec(
        side=side,
        trade_type=trade_type,
        exchange_pair=ExchangePair(exchange, USD / BTC)
    )

    assert order_spec.id
    assert order_spec.side == side
    assert order_spec.type == trade_type
    assert order_spec.exchange_pair == ExchangePair(exchange, USD / BTC)
    assert not order_spec.criteria

    # Create order specification with criteria
    order_spec = OrderSpec(
        side=side,
        trade_type=trade_type,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        criteria=lambda order, exchange: True
    )

    assert order_spec.id
    assert order_spec.side == side
    assert order_spec.type == trade_type
    assert order_spec.exchange_pair == ExchangePair(exchange, USD / BTC)
    assert order_spec.criteria


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Order")
def test_create_from_buy_order(mock_order_class,
                               mock_exchange_class):
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 2 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = mock_order_class.return_value
    order.portfolio = portfolio
    order.exchange_pair = ExchangePair(exchange, USD / BTC)
    order.path_id = "fake_path_id"
    order.price = Decimal(7000.00)

    wallet_btc = portfolio.get_wallet(exchange.id, BTC)
    wallet_btc.lock(
        quantity=0.4 * BTC,
        order=order,
        reason="test"
    )

    assert float(wallet_btc.balance.size) == 1.6
    assert float(wallet_btc.locked[order.path_id].size) == 0.4

    order_spec = OrderSpec(
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC)
    )

    next_order = order_spec.create_order(order)
    assert next_order

    assert next_order.side == TradeSide.SELL
    assert next_order.type == TradeType.MARKET
    assert next_order.exchange_pair == ExchangePair(exchange, USD / BTC)
    assert next_order.path_id == order.path_id
    assert next_order.quantity.path_id == order.path_id
    assert next_order.quantity.instrument == BTC


@mock.patch("tensortrade.exchanges.Exchange")
@mock.patch("tensortrade.orders.Order")
def test_create_from_sell_order(mock_order_class,
                                mock_exchange_class):
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 2 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = mock_order_class.return_value
    order.portfolio = portfolio
    order.exchange_pair = ExchangePair(exchange, USD / BTC)
    order.path_id = "fake_path_id"
    order.price = 7000.00

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    wallet_usd.lock(
        quantity=1000 * USD,
        order=order,
        reason="test"
    )

    assert float(wallet_usd.balance.size) == 9000
    assert float(wallet_usd.locked[order.path_id].size) == 1000

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC)
    )

    next_order = order_spec.create_order(order)
    assert next_order

    assert next_order.side == TradeSide.BUY
    assert next_order.type == TradeType.MARKET
    assert next_order.exchange_pair == ExchangePair(exchange, USD / BTC)
    assert next_order.path_id == order.path_id
    assert next_order.quantity.path_id == order.path_id
    assert next_order.quantity.instrument == USD


@mock.patch("tensortrade.exchanges.Exchange")
def test_to_dict(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC)
    )

    d = order_spec.to_dict()
    assert d == {
        "id": order_spec.id,
        "type": order_spec.type,
        "exchange_pair": order_spec.exchange_pair,
        "criteria": order_spec.criteria
    }

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        criteria=lambda order, exchange: True
    )

    d = order_spec.to_dict()
    assert d == {
        "id": order_spec.id,
        "type": order_spec.type,
        "exchange_pair": order_spec.exchange_pair,
        "criteria": order_spec.criteria
    }


@mock.patch("tensortrade.exchanges.Exchange")
def test_str(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "bitfinex"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = mock.Mock(return_value=Decimal(7000.00))

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC)
    )

    pattern = re.compile("<[A-Z][a-zA-Z]*:\\s(\\w+=.*,\\s)*(\\w+=.*)>")

    string = str(order_spec)
    assert string

    assert string == pattern.fullmatch(string).string
