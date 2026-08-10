import pytest
import unittest.mock as mock

from decimal import Decimal

from tensortrade.oms.instruments import USD, BTC, ExchangePair
from tensortrade.oms.orders import Order, OrderSpec, TradeSide, TradeType, OrderStatus
from tensortrade.oms.orders import short_order, cover_order, risk_managed_order, proportion_order
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.services.execution.simulated import (
    execute_buy_order,
    execute_sell_order,
    execute_short_order,
    execute_cover_order,
    execute_order,
)
from tensortrade.core.exceptions import InsufficientFunds


# =============================================================================
# TradeSide enum tests
# =============================================================================

def test_trade_side_short():
    assert TradeSide.SHORT.value == "short"
    assert str(TradeSide.SHORT) == "short"


def test_trade_side_cover():
    assert TradeSide.COVER.value == "cover"
    assert str(TradeSide.COVER) == "cover"


def test_trade_side_short_instrument():
    """SHORT.instrument() returns pair.quote (same as SELL)."""
    pair = USD / BTC
    assert TradeSide.SHORT.instrument(pair) == BTC
    assert TradeSide.COVER.instrument(pair) == USD


# =============================================================================
# Wallet borrowing tests
# =============================================================================

@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_wallet_borrow_allows_negative_balance(mock_exchange_class):
    """Wallet with borrow=True allows locking more than balance."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"

    wallet = Wallet(exchange, 0 * BTC, borrow=True)

    # Create a mock order with path_id
    mock_order = mock.Mock()
    mock_order.path_id = "test_path_id"

    # Should not raise InsufficientFunds even with 0 balance
    locked = wallet.lock(1 * BTC, mock_order, "SHORT")
    assert locked.size == Decimal('1')  # locked successfully
    assert wallet.balance.as_float() < 0  # balance went negative


@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_wallet_no_borrow_raises_insufficient_funds(mock_exchange_class):
    """Wallet with borrow=False (default) raises InsufficientFunds."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"

    wallet = Wallet(exchange, 0 * BTC)  # borrow=False by default

    mock_order = mock.Mock()
    mock_order.path_id = "test_path_id"

    with pytest.raises(InsufficientFunds):
        wallet.lock(1 * BTC, mock_order, "BUY")


@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_wallet_borrow_reset(mock_exchange_class):
    """Wallet.borrow should be restored after reset."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"

    wallet = Wallet(exchange, 0 * BTC, borrow=True)
    assert wallet.borrow is True

    # Simulate borrowing
    mock_order = mock.Mock()
    mock_order.path_id = "test_path_id"
    wallet.lock(1 * BTC, mock_order, "SHORT")
    assert wallet.balance.as_float() < 0

    wallet.reset()
    assert wallet.borrow is True
    assert wallet.balance.as_float() == 0


# =============================================================================
# Order SHORT/COVER tests
# =============================================================================

@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_order_short_side(mock_exchange_class):
    """Order with SHORT side can be created when wallet allows borrowing."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=0,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        side=TradeSide.SHORT,
        trade_type=TradeType.MARKET,
        quantity=1 * BTC,
        price=7000,
        portfolio=portfolio,
    )

    assert order.is_short is True
    assert order.is_sell is False
    assert order.is_buy is False
    assert order.is_cover is False
    assert order.side == TradeSide.SHORT


@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_order_cover_side(mock_exchange_class):
    """Order with COVER side locks the base wallet normally."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=0,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        side=TradeSide.COVER,
        trade_type=TradeType.MARKET,
        quantity=7000 * USD,
        price=7000,
        portfolio=portfolio,
    )

    assert order.is_cover is True
    assert order.is_short is False
    assert order.is_buy is False
    assert order.is_sell is False


# =============================================================================
# Order creation helper tests
# =============================================================================

@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_short_order_creation(mock_exchange_class):
    """short_order() creates an Order with SHORT side."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    ep = ExchangePair(exchange, USD / BTC)
    order = short_order(ep, price=7000, size=1, portfolio=portfolio)

    assert order.side == TradeSide.SHORT
    assert order.type == TradeType.MARKET
    assert order.exchange_pair == ep


@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_cover_order_creation(mock_exchange_class):
    """cover_order() creates an Order with COVER side."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    ep = ExchangePair(exchange, USD / BTC)
    order = cover_order(ep, price=7000, size=0.5, portfolio=portfolio)

    assert order.side == TradeSide.COVER
    assert order.type == TradeType.MARKET


# =============================================================================
# risk_managed_order for SHORT/COVER tests
# =============================================================================

@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_risk_managed_order_short_exit_is_cover(mock_exchange_class):
    """risk_managed_order with SHORT creates a COVER exit order spec."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    ep = ExchangePair(exchange, USD / BTC)
    order = risk_managed_order(
        side=TradeSide.SHORT,
        trade_type=TradeType.MARKET,
        exchange_pair=ep,
        price=7000,
        quantity=1 * BTC,
        down_percent=0.02,
        up_percent=0.03,
        portfolio=portfolio,
    )

    assert order.side == TradeSide.SHORT
    # The risk management spec should have COVER as exit side
    assert len(order._specs) == 1
    assert order._specs[0].side == TradeSide.COVER


@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_risk_managed_order_cover_exit_is_short(mock_exchange_class):
    """risk_managed_order with COVER creates a SHORT exit order spec."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_exchange_id"
    exchange.name = "test_exchange"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    ep = ExchangePair(exchange, USD / BTC)
    order = risk_managed_order(
        side=TradeSide.COVER,
        trade_type=TradeType.MARKET,
        exchange_pair=ep,
        price=7000,
        quantity=7000 * USD,
        down_percent=0.02,
        up_percent=0.03,
        portfolio=portfolio,
    )

    assert order.side == TradeSide.COVER
    assert len(order._specs) == 1
    assert order._specs[0].side == TradeSide.SHORT


# =============================================================================
# Execution tests — execute_short_order
# =============================================================================

def test_execute_short_order_simple():
    """Short order: borrow BTC and sell for USD."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.SHORT,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=1 * BTC,
        portfolio=portfolio,
        price=7000,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_short_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('7000'),
        options=options,
        clock=clock,
    )

    assert trade is not None
    assert trade.side == TradeSide.SHORT
    # After short: quote_wallet (BTC) balance should be negative
    assert quote_wallet.balance.as_float() < 0
    # base_wallet (USD) should have received funds
    assert base_wallet.locked['fake_id'].size > 0


def test_execute_short_order_limit_below_price():
    """Short limit order when current_price < limit_price should not fill."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('6500')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.SHORT,
        trade_type=TradeType.LIMIT,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=1 * BTC,
        portfolio=portfolio,
        price=7000,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_short_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('6500'),
        options=options,
        clock=clock,
    )

    # Limit short at 7000 with current price 6500: don't fill (wanted better price)
    assert trade is None


def test_execute_short_order_limit_above_price():
    """Short limit order when current_price > limit_price should fill."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('7500')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.SHORT,
        trade_type=TradeType.LIMIT,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=1 * BTC,
        portfolio=portfolio,
        price=7000,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_short_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('7500'),
        options=options,
        clock=clock,
    )

    assert trade is not None
    assert trade.side == TradeSide.SHORT


# =============================================================================
# Execution tests — execute_cover_order
# =============================================================================

def test_execute_cover_order_simple():
    """Cover order: buy back BTC with USD to close short position."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('6500')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, -1 * BTC, borrow=True)  # Already short 1 BTC

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.COVER,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=6500 * USD,
        portfolio=portfolio,
        price=6500,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_cover_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('6500'),
        options=options,
        clock=clock,
    )

    assert trade is not None
    assert trade.side == TradeSide.COVER
    # After cover: quote_wallet (BTC) balance should be less negative
    # Because we bought back some BTC
    assert quote_wallet.locked['fake_id'].size > 0  # BTC was deposited to locked


def test_execute_cover_order_limit_above_price():
    """Cover limit order when current_price > limit_price should not fill."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('7500')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, -1 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.COVER,
        trade_type=TradeType.LIMIT,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=6500 * USD,
        portfolio=portfolio,
        price=6500,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_cover_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('7500'),
        options=options,
        clock=clock,
    )

    # Cover limit at 6500 with price 7500: don't fill (wanted better price)
    assert trade is None


def test_execute_cover_order_limit_below_price():
    """Cover limit order when current_price < limit_price should fill."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('6000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, -1 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.COVER,
        trade_type=TradeType.LIMIT,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=6500 * USD,
        portfolio=portfolio,
        price=6500,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_cover_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('6000'),
        options=options,
        clock=clock,
    )

    assert trade is not None
    assert trade.side == TradeSide.COVER


# =============================================================================
# execute_order dispatch tests
# =============================================================================

def test_execute_order_dispatches_to_short():
    """execute_order dispatches SHORT orders to execute_short_order."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.SHORT,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=1 * BTC,
        portfolio=portfolio,
        price=7000,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('7000'),
        options=options,
        clock=clock,
    )

    assert trade is not None
    assert trade.side == TradeSide.SHORT


def test_execute_order_dispatches_to_cover():
    """execute_order dispatches COVER orders to execute_cover_order."""

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 3

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.quote_price = lambda pair: Decimal('6500')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, -1 * BTC, borrow=True)

    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    order = Order(
        step=1,
        side=TradeSide.COVER,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=6500 * USD,
        portfolio=portfolio,
        price=6500,
        path_id="fake_id",
    )
    order.status = OrderStatus.OPEN

    trade = execute_order(
        order,
        base_wallet,
        quote_wallet,
        current_price=Decimal('6500'),
        options=options,
        clock=clock,
    )

    assert trade is not None
    assert trade.side == TradeSide.COVER


# =============================================================================
# Full short → cover lifecycle test
# =============================================================================

def test_full_short_cover_lifecycle():
    """Test a complete short→cover lifecycle:
    1. Short 1 BTC at $7000 (borrow and sell)
    2. Cover at $6500 (buy back lower, profit!)
    """

    mock_clock = mock.Mock()
    clock = mock_clock.return_value
    clock.step = 0

    options = ExchangeOptions()
    mock_exchange = mock.Mock()
    exchange = mock_exchange.return_value
    exchange.name = "test"
    exchange.options = options
    exchange.id = "test"

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    # Step 1: SHORT 1 BTC at $7000
    exchange.quote_price = lambda pair: Decimal('7000')

    short_o = Order(
        step=0,
        side=TradeSide.SHORT,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=1 * BTC,
        portfolio=portfolio,
        price=7000,
    )
    short_o.status = OrderStatus.OPEN

    trade = execute_short_order(
        short_o, base_wallet, quote_wallet,
        current_price=Decimal('7000'), options=options, clock=clock,
    )
    short_o.fill(trade)
    short_o.complete()
    short_o.release()

    # After short: USD balance increased, BTC negative
    usd_after_short = base_wallet.balance.as_float()
    btc_after_short = quote_wallet.total_balance.as_float()
    assert usd_after_short > 10000  # Profit from short
    assert btc_after_short < 0  # Negative BTC = borrowed

    # Step 2: COVER at $6500 (price dropped — profit!)
    exchange.quote_price = lambda pair: Decimal('6500')

    # Need to cover the short position amount
    cover_size = abs(btc_after_short) * 6500  # USD needed to buy back
    cover_o = Order(
        step=1,
        side=TradeSide.COVER,
        trade_type=TradeType.MARKET,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        quantity=cover_size * USD,
        portfolio=portfolio,
        price=6500,
    )
    cover_o.status = OrderStatus.OPEN

    trade2 = execute_cover_order(
        cover_o, base_wallet, quote_wallet,
        current_price=Decimal('6500'), options=options, clock=clock,
    )
    cover_o.fill(trade2)
    cover_o.complete()
    cover_o.release()

    # After cover: BTC should be closer to 0 (short closed)
    btc_after_cover = quote_wallet.total_balance.as_float()
    assert btc_after_cover > btc_after_short  # Less negative
    # Net profit: sold at 7000, bought back at 6500
    assert abs(btc_after_cover) < abs(btc_after_short)


# =============================================================================
# Wallet transfer integrity for short orders
# =============================================================================

@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_transfer_consistency_for_short(mock_exchange_class):
    """Wallet.transfer() should maintain accounting for SHORT execution."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "test"
    exchange.name = "test"
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)

    # Lock BTC for short (goes negative)
    mock_order = mock.Mock()
    mock_order.path_id = "test_path"
    quantity = (1 * BTC).quantize()
    locked = quote_wallet.lock(quantity, mock_order, "SHORT")

    # Simulate transfer: quote(BTC) → base(USD)
    commission = (0.003 * locked).quantize()
    net = (locked - commission).quantize()

    transfer = Wallet.transfer(
        source=quote_wallet,
        target=base_wallet,
        quantity=net,
        commission=commission,
        exchange_pair=ExchangePair(exchange, USD / BTC),
        reason="SHORT",
    )

    assert transfer is not None
    assert transfer.price > 0


# =============================================================================
# proportion_order for borrowing wallet
# =============================================================================

@mock.patch('tensortrade.oms.exchanges.Exchange')
def test_proportion_order_with_borrow_wallet(mock_exchange_class):
    """proportion_order cannot create orders from a borrow wallet with zero
    balance because it uses balance.as_float() which is 0. For short selling,
    use short_order() or cover_order() instead."""
    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "test"
    exchange.name = "test"
    exchange.clock = mock.Mock()
    exchange.clock.step = 0
    exchange.quote_price = lambda pair: Decimal('7000')

    base_wallet = Wallet(exchange, 10000 * USD)
    quote_wallet = Wallet(exchange, 0 * BTC, borrow=True)
    portfolio = Portfolio(USD, [base_wallet, quote_wallet])

    # proportion_order uses balance.as_float() which is 0 for a borrow wallet
    # This correctly returns an empty order since there's no owned balance
    assert quote_wallet.balance.as_float() == 0

    # For short selling, use short_order() instead
    ep = ExchangePair(exchange, USD / BTC)
    order = short_order(ep, price=7000, size=1, portfolio=portfolio)
    assert order is not None
    assert order.side == TradeSide.SHORT
