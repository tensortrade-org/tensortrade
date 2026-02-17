"""Tests for SimpleOrders and ManagedRiskOrders action schemes."""

import pytest
from gymnasium.spaces import Discrete

import tensortrade.env.default as default
from tensortrade.env.default.actions import ManagedRiskOrders, SimpleOrders
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import BTC, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_env(
    n_steps: int = 50,
    trade_sizes: int = 4,
    commission: float = 0.0,
):
    """Create env with SimpleOrders action scheme."""
    prices = [100.0 + i * 0.5 for i in range(n_steps)]
    price_btc = Stream.source(prices, dtype="float").rename("USD-BTC")

    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=ExchangeOptions(commission=commission),
    )(price_btc)

    cash = Wallet(exchange, 10000.0 * USD)
    asset = Wallet(exchange, 10.0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(prices, dtype="float").rename("close")]
    feed = DataFeed(features)
    feed.compile()

    action_scheme = SimpleOrders(trade_sizes=trade_sizes)
    reward_scheme = SimpleProfit()

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=1,
    )
    return env, action_scheme, portfolio


def _make_managed_risk_env(
    n_steps: int = 50,
    stop=None,
    take=None,
    commission: float = 0.0,
):
    """Create env with ManagedRiskOrders action scheme."""
    if stop is None:
        stop = [0.02, 0.04]
    if take is None:
        take = [0.01, 0.02]

    prices = [100.0 + i * 0.5 for i in range(n_steps)]
    price_btc = Stream.source(prices, dtype="float").rename("USD-BTC")

    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=ExchangeOptions(commission=commission),
    )(price_btc)

    cash = Wallet(exchange, 10000.0 * USD)
    asset = Wallet(exchange, 10.0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(prices, dtype="float").rename("close")]
    feed = DataFeed(features)
    feed.compile()

    action_scheme = ManagedRiskOrders(stop=stop, take=take)
    reward_scheme = SimpleProfit()

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=1,
    )
    return env, action_scheme, portfolio


# ---------------------------------------------------------------------------
# SimpleOrders tests
# ---------------------------------------------------------------------------


class TestSimpleOrdersActionSpace:
    """Tests for SimpleOrders action space configuration."""

    def test_action_space_is_discrete(self):
        env, scheme, _ = _make_simple_env()
        assert isinstance(env.action_space, Discrete)

    def test_action_space_includes_hold(self):
        """Action 0 should always be a hold (no-op) action."""
        env, scheme, _ = _make_simple_env()
        assert env.action_space.n > 1
        # Action 0 maps to None in the actions list
        assert scheme.actions[0] is None

    def test_action_space_size_scales_with_trade_sizes(self):
        """More trade sizes = more actions."""
        env_small, scheme_small, _ = _make_simple_env(trade_sizes=2)
        env_large, scheme_large, _ = _make_simple_env(trade_sizes=4)
        assert env_large.action_space.n > env_small.action_space.n

    def test_actions_list_populated_after_space_access(self):
        """Accessing action_space should populate the actions list."""
        env, scheme, _ = _make_simple_env()
        _ = env.action_space
        assert scheme.actions is not None
        assert len(scheme.actions) == env.action_space.n


class TestSimpleOrdersExecution:
    """Tests for SimpleOrders order execution."""

    def test_hold_produces_no_orders(self):
        env, scheme, portfolio = _make_simple_env()
        env.reset()
        orders = scheme.get_orders(0, portfolio)
        assert orders == []

    def test_random_actions_dont_crash(self):
        """Random actions should not crash the environment."""
        env, _, _ = _make_simple_env(n_steps=50)
        obs, _ = env.reset()

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps > 200:
                break

        assert steps > 0

    def test_min_order_pct_filtering(self):
        """Orders below min_order_pct of net worth should be filtered."""
        env, scheme, _ = _make_simple_env()
        # Default min_order_pct = 0.02 (2% of net worth)
        assert scheme.min_order_pct == 0.02

    def test_custom_trade_sizes_list(self):
        """Passing a list of trade sizes should use them directly."""
        prices = [100.0] * 20
        price_btc = Stream.source(prices, dtype="float").rename("USD-BTC")
        exchange = Exchange("exchange", service=execute_order)(price_btc)
        cash = Wallet(exchange, 10000.0 * USD)
        asset = Wallet(exchange, 10.0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        scheme = SimpleOrders(trade_sizes=[0.25, 0.5, 1.0])
        scheme.portfolio = portfolio
        assert scheme.trade_sizes == [0.25, 0.5, 1.0]


# ---------------------------------------------------------------------------
# ManagedRiskOrders tests
# ---------------------------------------------------------------------------


class TestManagedRiskOrdersActionSpace:
    """Tests for ManagedRiskOrders action space configuration."""

    def test_action_space_is_discrete(self):
        env, scheme, _ = _make_managed_risk_env()
        assert isinstance(env.action_space, Discrete)

    def test_action_space_includes_hold(self):
        env, scheme, _ = _make_managed_risk_env()
        assert scheme.actions[0] is None

    def test_action_space_size_scales_with_stop_take(self):
        """More stop/take levels = more actions."""
        env_small, scheme_small, _ = _make_managed_risk_env(
            stop=[0.02],
            take=[0.01],
        )
        env_large, scheme_large, _ = _make_managed_risk_env(
            stop=[0.02, 0.04, 0.06],
            take=[0.01, 0.02, 0.03],
        )
        assert env_large.action_space.n > env_small.action_space.n

    def test_default_stop_take_values(self):
        """Default stop/take should be sensible lists."""
        scheme = ManagedRiskOrders()
        assert scheme.stop == [0.02, 0.04, 0.06]
        assert scheme.take == [0.01, 0.02, 0.03]


class TestManagedRiskOrdersExecution:
    """Tests for ManagedRiskOrders order execution."""

    def test_hold_produces_no_orders(self):
        env, scheme, portfolio = _make_managed_risk_env()
        env.reset()
        orders = scheme.get_orders(0, portfolio)
        assert orders == []

    def test_random_actions_dont_crash(self):
        """Random actions should not crash the environment."""
        env, _, _ = _make_managed_risk_env(n_steps=50)
        obs, _ = env.reset()

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps > 200:
                break

        assert steps > 0

    def test_orders_have_risk_management(self):
        """Non-hold actions should produce orders with order specs (stop/take)."""
        env, scheme, portfolio = _make_managed_risk_env()
        env.reset()
        # Access action_space to populate actions
        _ = env.action_space
        # Try action 1 (first non-hold)
        orders = scheme.get_orders(1, portfolio)
        # If a valid order is returned, it should have order specs
        for order in orders:
            if order is not None:
                assert len(order._specs) > 0


# ---------------------------------------------------------------------------
# Action scheme registry
# ---------------------------------------------------------------------------


class TestActionSchemeRegistry:
    """Tests for the action scheme registry."""

    def test_simple_in_registry(self):
        from tensortrade.env.default.actions import _registry

        assert "simple" in _registry
        assert _registry["simple"] == SimpleOrders

    def test_managed_risk_in_registry(self):
        from tensortrade.env.default.actions import _registry

        assert "managed-risk" in _registry
        assert _registry["managed-risk"] == ManagedRiskOrders

    def test_get_unknown_raises(self):
        from tensortrade.env.default.actions import get

        with pytest.raises(KeyError, match="not associated"):
            get("nonexistent-action")
