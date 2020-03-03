

import pytest
import unittest.mock as mock

from tensortrade.wallets import Wallet, Portfolio
from tensortrade.instruments import BTC, USD, ExchangePair, precise
from tensortrade.orders import Order, TradeSide, TradeType, OrderStatus
from tensortrade.exchanges import Exchange, ExchangeOptions

from tensortrade.exchanges.services.execution.simulated import execute_buy_order, execute_sell_order


@mock.patch('tensortrade.base.Clock')
@mock.patch('tensortrade.exchanges.Exchange')
def test_simple_values_execute_buy_order(mock_exchange, mock_clock):

    clock = mock_clock.return_value
    clock.step = 3

    # Test: current_price > order.price
    current_price = 9600
    options = ExchangeOptions()
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, 10000*USD)
    quote_wallet = Wallet(exchange, 1.7*BTC)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD/BTC),
        quantity=3000 * USD,
        portfolio=portfolio,
        price=9200,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_buy_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    base_balance = base_wallet.locked['fake_id'].size
    quote_balance = quote_wallet.locked['fake_id'].size

    assert base_balance == 3000 - trade.size + trade.commission.size
    assert quote_balance == trade.size / current_price

    # Test: current_price < order.price
    current_price = 9000
    options = ExchangeOptions()
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 1.7 * BTC)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=3000 * USD,
        portfolio=portfolio,
        price=9200,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_buy_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    base_balance = base_wallet.locked['fake_id'].size
    quote_balance = quote_wallet.locked['fake_id'].size

    assert base_balance == 3000 - trade.size + trade.commission.size
    assert quote_balance == trade.size / current_price


@mock.patch('tensortrade.base.Clock')
@mock.patch('tensortrade.exchanges.Exchange')
def test_simple_values_execute_sell_order(mock_exchange, mock_clock):

    clock = mock_clock.return_value
    clock.step = 3

    # Test: current_price < order.price
    current_price = 9000
    options = ExchangeOptions()
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 1.7 * BTC)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD/BTC),
        quantity=0.5 * BTC,
        portfolio=portfolio,
        price=9200,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_sell_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    base_balance = base_wallet.locked['fake_id'].size
    quote_balance = quote_wallet.locked['fake_id'].size

    assert base_balance == trade.size
    assert quote_balance == 0.5 - ((trade.size + trade.commission.size) / current_price)

    # Test: current_price > order.price
    current_price = 9600
    options = ExchangeOptions()
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, 10000*USD)
    quote_wallet = Wallet(exchange, 1.7*BTC)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD/BTC),
        quantity=0.5 * BTC,
        portfolio=portfolio,
        price=9200,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_sell_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    base_balance = base_wallet.locked['fake_id'].size
    quote_balance = quote_wallet.locked['fake_id'].size

    assert base_balance == trade.size
    assert quote_balance == 0.5 - ((trade.size + trade.commission.size) / current_price)


@mock.patch('tensortrade.base.Clock')
@mock.patch('tensortrade.exchanges.Exchange')
def test_complex_values_execute_buy_order(mock_exchange, mock_clock):

    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()

    # =================================
    # Test: current_price < order.price
    # =================================
    base_balance = 2298449.19 * USD
    quote_balance = 2.39682929 * BTC
    order_quantity = 56789.33 * USD
    order_price = 9588.66

    current_price = 9678.43
    # =================================

    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, base_balance)
    quote_wallet = Wallet(exchange, quote_balance)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD/BTC),
        quantity=order_quantity,
        portfolio=portfolio,
        price=order_price,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_buy_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    locked_base_balance = base_wallet.locked['fake_id'].size
    locked_quote_balance = quote_wallet.locked['fake_id'].size

    assert locked_base_balance == order_quantity.size - (trade.size + trade.commission.size)
    assert locked_quote_balance == trade.size / current_price

    # =================================
    # Test: current_price > order.price
    # =================================
    order_price = 10819.66

    current_price = 9678.43
    # =================================
    current_price = 9000
    options = ExchangeOptions()
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, base_balance)
    quote_wallet = Wallet(exchange, quote_balance)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=order_quantity,
        portfolio=portfolio,
        price=order_price,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_buy_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    locked_base_balance = base_wallet.locked['fake_id'].size
    locked_quote_balance = quote_wallet.locked['fake_id'].size

    assert locked_base_balance == order_quantity.size - (trade.size + trade.commission.size)
    assert locked_quote_balance == trade.size / current_price


@mock.patch('tensortrade.base.Clock')
@mock.patch('tensortrade.exchanges.Exchange')
def test_complex_values_execute_sell_order(mock_exchange, mock_clock):

    clock = mock_clock.return_value
    clock.step = 3

    # =================================
    # Test: current_price < order.price
    # =================================
    base_balance = 2298449.19 * USD
    quote_balance = 10.39682929 * BTC
    order_quantity = 3.33829407 * BTC
    order_price = 9588.66

    current_price = 9478.43
    # =================================

    options = ExchangeOptions()
    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, base_balance)
    quote_wallet = Wallet(exchange, quote_balance)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD/BTC),
        quantity=order_quantity,
        portfolio=portfolio,
        price=order_price,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_sell_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    locked_base_balance = base_wallet.locked['fake_id'].size
    locked_quote_balance = quote_wallet.locked['fake_id'].size

    assert locked_base_balance == trade.size
    assert locked_quote_balance == order_quantity.size - ((trade.size + trade.commission.size) / current_price)

    # =================================
    # Test: current_price > order.price
    # =================================
    current_price = 9678.43
    # =================================

    exchange = mock_exchange.return_value
    exchange.name = "coinbase"
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, base_balance)
    quote_wallet = Wallet(exchange, quote_balance)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD/BTC),
        quantity=order_quantity,
        portfolio=portfolio,
        price=order_price,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    trade = execute_sell_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=current_price,
        options=options,
        clock=clock,
    )

    locked_base_balance = base_wallet.locked['fake_id'].size
    locked_quote_balance = quote_wallet.locked['fake_id'].size

    assert locked_base_balance == trade.size
    assert locked_quote_balance == order_quantity.size - ((trade.size + trade.commission.size) / current_price)
