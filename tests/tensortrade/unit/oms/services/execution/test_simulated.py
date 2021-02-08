import unittest.mock as mock

from decimal import Decimal

from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import BTC, USD, ExchangePair
from tensortrade.oms.orders import Order, TradeSide, TradeType, OrderStatus
from tensortrade.oms.exchanges import ExchangeOptions

from tensortrade.oms.services.execution.simulated import execute_buy_order, execute_sell_order


def assert_execute_order(current_price,
                         base_balance,
                         quote_balance,
                         order_side,
                         order_quantity,
                         order_price,
                         ):
    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    base = base_balance.instrument
    quote = quote_balance.instrument

    current_price = Decimal(current_price).quantize(Decimal(10) ** -base.precision)
    order_price = Decimal(order_price).quantize(Decimal(10) ** -base.precision)

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "bitfinex"
    exchange.options = options
    exchange.quote_price = lambda pair: current_price

    base_wallet = Wallet(exchange, base_balance)
    quote_wallet = Wallet(exchange, quote_balance)

    portfolio = Portfolio(USD, [
        base_wallet,
        quote_wallet
    ])

    order = Order(
        step=1,
        side=order_side,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, base/quote),
        quantity=order_quantity,
        portfolio=portfolio,
        price=order_price,
        path_id="fake_id"
    )
    order.status = OrderStatus.OPEN

    if order_side == TradeSide.BUY:

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

        expected_base_balance = order_quantity.size - (trade.size + trade.commission.size)
        expected_quote_balance = trade.size / current_price

        expected_base_balance = expected_base_balance.quantize(Decimal(10) ** -base.precision)
        expected_quote_balance = expected_quote_balance.quantize(Decimal(10) ** -quote.precision)

        assert base_balance == expected_base_balance
        assert quote_balance == expected_quote_balance

    else:
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

        expected_base_balance = trade.size * current_price
        expected_base_balance = expected_base_balance.quantize(Decimal(10)**-base.precision)

        assert base_balance == expected_base_balance
        assert quote_balance == 0


def test_simple_values_execute_buy_order():

    # Test: current_price < order_price
    assert_execute_order(
        current_price=9000,
        base_balance=10000 * USD,
        quote_balance=1.7 * BTC,
        order_side=TradeSide.BUY,
        order_quantity=3000 * USD,
        order_price=9200
    )

    # Test: current_price > order_price
    assert_execute_order(
        current_price=9600,
        base_balance=10000 * USD,
        quote_balance=1.7 * BTC,
        order_side=TradeSide.BUY,
        order_quantity=3000 * USD,
        order_price=9200
    )


def test_simple_values_execute_sell_order():

    # Test: current_price < order_price
    assert_execute_order(
        current_price=9000,
        base_balance=10000 * USD,
        quote_balance=1.7 * BTC,
        order_side=TradeSide.SELL,
        order_quantity=0.5 * BTC,
        order_price=9200
    )

    # Test: current_price > order_price
    assert_execute_order(
        current_price=9600,
        base_balance=10000 * USD,
        quote_balance=1.7 * BTC,
        order_side=TradeSide.SELL,
        order_quantity=0.5 * BTC,
        order_price=9200
    )


def test_complex_values_execute_buy_order():

    # Test: current_price < order_price
    assert_execute_order(
        current_price=9588.66,
        base_balance=2298449.19 * USD,
        quote_balance=2.39682929 * BTC,
        order_side=TradeSide.BUY,
        order_quantity=56789.33 * USD,
        order_price=9678.43
    )

    # Test: current_price > order_price
    assert_execute_order(
        current_price=10819.66,
        base_balance=2298449.19 * USD,
        quote_balance=2.39682929 * BTC,
        order_side=TradeSide.BUY,
        order_quantity=56789.33 * USD,
        order_price=9678.43
    )


def test_complex_values_execute_sell_order():

    # Test: current_price < order_price
    assert_execute_order(
        current_price=9588.66,
        base_balance=2298449.19 * USD,
        quote_balance=3.33829407 * BTC,
        order_side=TradeSide.SELL,
        order_quantity=2.39682929 * BTC,
        order_price=9678.43
    )

    # Test: current_price > order_price
    assert_execute_order(
        current_price=10819.66,
        base_balance=2298449.19 * USD,
        quote_balance=3.33829407 * BTC,
        order_side=TradeSide.SELL,
        order_quantity=2.39682929 * BTC,
        order_price=9678.43
    )
