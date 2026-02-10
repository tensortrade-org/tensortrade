"""Tests for Order lifecycle: execute, fill, complete, cancel, release, serialization."""

import unittest.mock as mock
from decimal import Decimal

import pytest

from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.instruments import BTC, USD, ExchangePair, TradingPair
from tensortrade.oms.orders import Order, OrderSpec, TradeSide, TradeType
from tensortrade.oms.orders.order import OrderStatus
from tensortrade.oms.wallets import Portfolio, Wallet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    side=TradeSide.BUY,
    trade_type=TradeType.MARKET,
    quantity_amount=5000,
    price=100.0,
    start=None,
    end=None,
):
    """Create an order with a mocked exchange."""
    exchange = mock.MagicMock()
    exchange.options = ExchangeOptions()
    exchange.id = "test_exchange"
    exchange.name = "test_exchange"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 10 * BTC)]
    portfolio = Portfolio(USD, wallets)

    ep = ExchangePair(exchange, USD / BTC)
    quantity = quantity_amount * USD

    order = Order(
        step=0,
        side=side,
        trade_type=trade_type,
        exchange_pair=ep,
        price=price,
        quantity=quantity,
        portfolio=portfolio,
        start=start,
        end=end,
    )
    return order, portfolio, exchange, wallets


# ---------------------------------------------------------------------------
# OrderStatus enum
# ---------------------------------------------------------------------------


class TestOrderStatus:
    """Tests for OrderStatus enum values."""

    def test_all_statuses_exist(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"

    def test_str_representation(self):
        assert str(OrderStatus.PENDING) == "pending"
        assert str(OrderStatus.FILLED) == "filled"
        assert str(OrderStatus.CANCELLED) == "cancelled"


# ---------------------------------------------------------------------------
# Order creation
# ---------------------------------------------------------------------------


class TestOrderCreation:
    """Tests for Order constructor and initial state."""

    def test_initial_status_is_pending(self):
        order, _, _, _ = _make_order()
        assert order.status == OrderStatus.PENDING

    def test_order_has_unique_path_id(self):
        order1, _, _, _ = _make_order()
        order2, _, _, _ = _make_order()
        assert order1.path_id != order2.path_id

    def test_order_properties_buy(self):
        order, _, _, _ = _make_order(side=TradeSide.BUY)
        assert order.is_buy
        assert not order.is_sell

    def test_buy_order_is_not_sell(self):
        order, _, _, _ = _make_order(side=TradeSide.BUY)
        assert not order.is_sell

    def test_market_order_type(self):
        order, _, _, _ = _make_order(trade_type=TradeType.MARKET)
        assert order.is_market_order
        assert not order.is_limit_order

    def test_limit_order_type(self):
        order, _, _, _ = _make_order(trade_type=TradeType.LIMIT)
        assert order.is_limit_order
        assert not order.is_market_order

    def test_order_is_active_initially(self):
        order, _, _, _ = _make_order()
        assert order.is_active

    def test_quantity_locked_in_wallet(self):
        order, _, _, wallets = _make_order()
        cash_wallet = wallets[0]  # USD wallet
        assert len(cash_wallet.locked) > 0

    def test_remaining_equals_quantity(self):
        order, _, _, _ = _make_order()
        assert order.remaining == order.quantity

    def test_zero_quantity_raises(self):
        from tensortrade.core.exceptions import InvalidOrderQuantity

        exchange = mock.MagicMock()
        exchange.options = ExchangeOptions()
        exchange.id = "test_exchange"
        exchange.name = "test_exchange"

        wallets = [Wallet(exchange, 10000 * USD)]
        portfolio = Portfolio(USD, wallets)
        ep = ExchangePair(exchange, USD / BTC)

        with pytest.raises(InvalidOrderQuantity):
            Order(
                step=0,
                side=TradeSide.BUY,
                trade_type=TradeType.MARKET,
                exchange_pair=ep,
                price=100.0,
                quantity=0 * USD,
                portfolio=portfolio,
            )

    def test_pair_property(self):
        order, _, _, _ = _make_order()
        assert isinstance(order.pair, TradingPair)
        assert order.pair.base == USD
        assert order.pair.quote == BTC

    def test_base_quote_instrument(self):
        order, _, _, _ = _make_order()
        assert order.base_instrument == USD
        assert order.quote_instrument == BTC


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------


class TestOrderExecution:
    """Tests for Order.execute()."""

    def test_execute_sets_status_to_open(self):
        order, _, _, _ = _make_order()
        order.execute()
        assert order.status == OrderStatus.OPEN

    def test_execute_calls_exchange_execute_order(self):
        order, portfolio, exchange, _ = _make_order()
        order.execute()
        exchange.execute_order.assert_called_once_with(order, portfolio)

    def test_execute_notifies_listeners(self):
        order, _, _, _ = _make_order()
        listener = mock.MagicMock()
        order.attach(listener)
        order.execute()
        listener.on_execute.assert_called_once_with(order)

    def test_execute_attaches_portfolio_listener(self):
        order, portfolio, _, _ = _make_order()
        mock_listener = mock.MagicMock()
        portfolio.order_listener = mock_listener
        order.execute()
        # Should have attached the portfolio listener
        assert mock_listener in order.listeners


# ---------------------------------------------------------------------------
# Order cancellation
# ---------------------------------------------------------------------------


class TestOrderCancellation:
    """Tests for Order.cancel()."""

    def test_cancel_sets_status_to_cancelled(self):
        order, _, _, _ = _make_order()
        order.cancel("test cancel")
        assert order.status == OrderStatus.CANCELLED

    def test_is_cancelled_property(self):
        order, _, _, _ = _make_order()
        order.cancel()
        assert order.is_cancelled

    def test_cancel_notifies_listeners(self):
        order, _, _, _ = _make_order()
        listener = mock.MagicMock()
        order.attach(listener)
        order.cancel()
        listener.on_cancel.assert_called_once_with(order)

    def test_cancel_clears_listeners(self):
        order, _, _, _ = _make_order()
        listener = mock.MagicMock()
        order.attach(listener)
        order.cancel()
        assert len(order.listeners) == 0

    def test_cancelled_order_is_not_active(self):
        order, _, _, _ = _make_order()
        order.cancel()
        assert not order.is_active

    def test_cancelled_order_is_complete(self):
        order, _, _, _ = _make_order()
        order.cancel()
        assert order.is_complete

    def test_cancel_releases_locked_funds(self):
        order, _, _, wallets = _make_order()
        cash_wallet = wallets[0]
        # Before cancel, funds are locked
        assert order.path_id in cash_wallet.locked
        order.cancel()
        # After cancel, locked should be empty
        assert order.path_id not in cash_wallet.locked


# ---------------------------------------------------------------------------
# Order completion
# ---------------------------------------------------------------------------


class TestOrderCompletion:
    """Tests for Order.complete()."""

    def test_complete_sets_status_to_filled(self):
        order, _, _, _ = _make_order()
        order.execute()
        order.complete()
        assert order.status == OrderStatus.FILLED

    def test_complete_notifies_listeners(self):
        order, _, _, _ = _make_order()
        listener = mock.MagicMock()
        order.attach(listener)
        order.execute()
        order.complete()
        listener.on_complete.assert_called_once_with(order)

    def test_complete_clears_listeners(self):
        order, _, _, _ = _make_order()
        listener = mock.MagicMock()
        order.attach(listener)
        order.execute()
        order.complete()
        assert len(order.listeners) == 0

    def test_filled_order_is_not_active(self):
        order, _, _, _ = _make_order()
        order.execute()
        order.complete()
        assert not order.is_active


# ---------------------------------------------------------------------------
# Order add_order_spec
# ---------------------------------------------------------------------------


class TestAddOrderSpec:
    """Tests for Order.add_order_spec()."""

    def test_add_spec_returns_self(self):
        order, _, exchange, _ = _make_order()
        ep = ExchangePair(exchange, USD / BTC)
        spec = OrderSpec(
            side=TradeSide.SELL,
            trade_type=TradeType.MARKET,
            exchange_pair=ep,
            criteria=None,
        )
        result = order.add_order_spec(spec)
        assert result is order

    def test_add_spec_accumulates(self):
        order, _, exchange, _ = _make_order()
        ep = ExchangePair(exchange, USD / BTC)
        for _ in range(3):
            spec = OrderSpec(
                side=TradeSide.SELL,
                trade_type=TradeType.MARKET,
                exchange_pair=ep,
                criteria=None,
            )
            order.add_order_spec(spec)
        assert len(order._specs) == 3

    def test_complete_pops_spec(self):
        """Completing an order with specs should pop and try to create next order."""
        order, _, exchange, _ = _make_order()
        ep = ExchangePair(exchange, USD / BTC)
        spec = OrderSpec(
            side=TradeSide.SELL,
            trade_type=TradeType.MARKET,
            exchange_pair=ep,
            criteria=None,
        )
        order.add_order_spec(spec)
        assert len(order._specs) == 1

        order.execute()
        order.complete()
        # Spec should have been popped
        assert len(order._specs) == 0


# ---------------------------------------------------------------------------
# Order expiration
# ---------------------------------------------------------------------------


class TestOrderExpiration:
    """Tests for Order.is_expired property."""

    def test_not_expired_when_no_end(self):
        order, _, _, _ = _make_order(end=None)
        assert not order.is_expired

    def test_expired_when_clock_past_end(self):
        order, _, exchange, _ = _make_order(end=5)
        exchange.clock.step = 5
        assert order.is_expired

    def test_not_expired_when_clock_before_end(self):
        order, _, exchange, _ = _make_order(end=10)
        exchange.clock.step = 3
        assert not order.is_expired


# ---------------------------------------------------------------------------
# Order serialization
# ---------------------------------------------------------------------------


class TestOrderSerialization:
    """Tests for Order.to_dict() and to_json()."""

    def test_to_dict_has_required_keys(self):
        order, _, _, _ = _make_order()
        d = order.to_dict()
        required_keys = {
            "id",
            "step",
            "exchange_pair",
            "status",
            "type",
            "side",
            "quantity",
            "size",
            "remaining",
            "price",
            "criteria",
            "path_id",
            "created_at",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_values(self):
        order, _, _, _ = _make_order(price=100.0)
        d = order.to_dict()
        assert d["step"] == 0
        assert d["price"] == 100.0
        assert d["status"] == OrderStatus.PENDING

    def test_to_json_all_serializable(self):
        """to_json should return JSON-serializable values."""
        order, _, _, _ = _make_order()
        j = order.to_json()
        for key, value in j.items():
            assert isinstance(value, (str, int, float)), f"{key} is {type(value)}"

    def test_to_json_side_is_string(self):
        order, _, _, _ = _make_order()
        j = order.to_json()
        assert isinstance(j["side"], str)

    def test_str_representation(self):
        order, _, _, _ = _make_order()
        s = str(order)
        assert "Order" in s

    def test_repr_equals_str(self):
        order, _, _, _ = _make_order()
        assert repr(order) == str(order)

    def test_size_property(self):
        order, _, _, _ = _make_order(quantity_amount=5000)
        assert order.size == Decimal("5000")

    def test_no_trades_initially(self):
        order, _, _, _ = _make_order()
        assert order.trades == []


# ---------------------------------------------------------------------------
# Order release
# ---------------------------------------------------------------------------


class TestOrderRelease:
    """Tests for Order.release()."""

    def test_release_unlocks_funds(self):
        order, _, _, wallets = _make_order()
        cash_wallet = wallets[0]
        assert order.path_id in cash_wallet.locked

        order.release("test release")
        assert order.path_id not in cash_wallet.locked
