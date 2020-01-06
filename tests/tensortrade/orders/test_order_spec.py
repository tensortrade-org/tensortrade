
import re
import unittest.mock as mock

from tensortrade.orders import OrderSpec
from tensortrade.trades import TradeSide, TradeType
from tensortrade.instruments import USD, BTC, Quantity
from tensortrade.wallets import Portfolio, Wallet


def test_init():

    side = TradeSide.BUY
    trade_type = TradeType.MARKET
    pair = USD/BTC

    # Create order specification without criteria
    order_spec = OrderSpec(
        side=side,
        trade_type=trade_type,
        pair=pair
    )

    assert order_spec.id
    assert order_spec.side == side
    assert order_spec.type == trade_type
    assert order_spec.pair == pair
    assert not order_spec.criteria

    # Create order specification with criteria
    order_spec = OrderSpec(
        side=side,
        trade_type=trade_type,
        pair=pair,
        criteria=lambda order, exchange: True
    )

    assert order_spec.id
    assert order_spec.side == side
    assert order_spec.type == trade_type
    assert order_spec.pair == pair
    assert order_spec.criteria


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_create_from_buy_order(mock_order_class,
                               mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_id"

    wallets = [Wallet(exchange, 10000.00*USD), Wallet(exchange, 2*BTC)]
    portfolio = Portfolio(USD, wallets)

    order = mock_order_class.return_value
    order.portfolio = portfolio
    order.path_id = "fake_path_id"
    order.price = 7000.00

    wallet_btc = portfolio.get_wallet(exchange.id, BTC)
    wallet_btc -= 0.4*BTC
    wallet_btc += Quantity(BTC, 0.4, path_id=order.path_id)

    assert wallet_btc.balance == 1.6 * BTC
    assert wallet_btc.locked[order.path_id].size == 0.4 * BTC

    order_spec = OrderSpec(
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        pair=USD/BTC
    )

    next_order = order_spec.create_order(order, exchange)
    assert next_order

    assert next_order.side == TradeSide.SELL
    assert next_order.type == TradeType.MARKET
    assert next_order.pair == USD/BTC
    assert next_order.path_id == order.path_id
    assert next_order.quantity.path_id == order.path_id
    assert next_order.quantity.instrument == BTC


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Order')
def test_create_from_sell_order(mock_order_class,
                                mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_id"

    wallets = [Wallet(exchange, 10000.00*USD), Wallet(exchange, 2*BTC)]
    portfolio = Portfolio(USD, wallets)

    order = mock_order_class.return_value
    order.portfolio = portfolio
    order.path_id = "fake_path_id"
    order.price = 7000.00

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    wallet_usd -= 1000*USD
    wallet_usd += Quantity(USD, 1000, path_id=order.path_id)

    assert wallet_usd.balance == 9000 * USD
    assert wallet_usd.locked[order.path_id].size == 1000 * USD

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        pair=USD/BTC
    )

    next_order = order_spec.create_order(order, exchange)
    assert next_order

    assert next_order.side == TradeSide.BUY
    assert next_order.type == TradeType.MARKET
    assert next_order.pair == USD/BTC
    assert next_order.path_id == order.path_id
    assert next_order.quantity.path_id == order.path_id
    assert next_order.quantity.instrument == USD


def test_to_dict():

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        pair=USD / BTC
    )

    d = order_spec.to_dict()
    assert d == {
        "id": order_spec.id,
        "type": order_spec.type,
        "pair": order_spec.pair,
        "criteria": order_spec.criteria
    }

    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        pair=USD / BTC,
        criteria=lambda order, exchange: True
    )

    d = order_spec.to_dict()
    assert d == {
        "id": order_spec.id,
        "type": order_spec.type,
        "pair": order_spec.pair,
        "criteria": order_spec.criteria
    }


def test_str():
    order_spec = OrderSpec(
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        pair=USD / BTC
    )

    pattern = re.compile("<[A-Z][a-zA-Z]*:\\s(\\w+=.*,\\s)*(\\w+=.*)>")

    string = str(order_spec)
    assert string

    assert string == pattern.fullmatch(string).string
