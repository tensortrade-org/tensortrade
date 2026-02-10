"""Extended tests for reward schemes: PBR stats, AdaptiveProfitSeeker, registry."""

from collections import OrderedDict

import pytest

import tensortrade.env.default as default
import tensortrade.env.default.rewards as rewards
from tensortrade.env.default.actions import BSH
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import BTC, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    reward_cls,
    prices=None,
    commission=0.0,
    initial_cash=10000.0,
    **reward_kwargs,
):
    """Create a minimal env with a given reward scheme class."""
    if prices is None:
        prices = [100.0 + i * 0.5 for i in range(20)]

    price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")
    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=ExchangeOptions(commission=commission),
    )(price_stream)

    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(prices, dtype="float").rename("close")]
    feed = DataFeed(features)
    feed.compile()

    # Some schemes need price stream (PBR, AdvancedPBR, AdaptiveProfitSeeker)
    if reward_cls in (
        rewards.PBR,
        rewards.AdvancedPBR,
        rewards.FractionalPBR,
        rewards.AdaptiveProfitSeeker,
    ):
        scheme = reward_cls(price=price_stream, commission=commission, **reward_kwargs)
    else:
        scheme = reward_cls(**reward_kwargs)

    action_scheme = BSH(cash=cash, asset=asset).attach(scheme)
    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=scheme,
        window_size=1,
        max_allowed_loss=0.99,
    )
    env.reset()
    return env, scheme


# ---------------------------------------------------------------------------
# PBR stats and churn logic
# ---------------------------------------------------------------------------


class TestPBRStats:
    """Thorough tests for PBR.get_stats() and churn penalty logic."""

    def test_stats_initial_state(self):
        """Stats should be all zeros before any actions."""
        _, scheme = _make_env(rewards.PBR)
        stats = scheme.get_stats()
        assert stats["trade_count"] == 0
        assert stats["buy_count"] == 0
        assert stats["sell_count"] == 0
        assert stats["hold_count"] == 0
        assert stats["churn_trade_count"] == 0
        assert stats["churn_ratio"] == 0.0
        assert stats["avg_penalty_per_trade"] == 0.0
        assert stats["total_commission_penalty"] == 0.0
        assert stats["cumulative_reward"] == 0.0

    def test_stats_after_buy_hold_sell(self):
        """Stats should track buy, hold, and sell correctly."""
        env, scheme = _make_env(rewards.PBR, commission=0.001)
        env.step(1)  # buy
        env.step(0)  # hold
        env.step(0)  # hold
        env.step(2)  # sell

        stats = scheme.get_stats()
        assert stats["buy_count"] == 1
        assert stats["sell_count"] == 1
        assert stats["hold_count"] == 2
        assert stats["trade_count"] == 2

    def test_churn_detection(self):
        """Rapid buy-sell within churn window should trigger churn count."""
        prices = [100.0] * 20
        env, scheme = _make_env(
            rewards.PBR,
            prices=prices,
            commission=0.01,
            churn_window=3,
        )

        env.step(1)  # buy  (step 1)
        env.step(2)  # sell (step 2, within 3 of last)
        env.step(1)  # buy  (step 3, within 3 of step 2)

        stats = scheme.get_stats()
        assert stats["churn_trade_count"] >= 1

    def test_no_churn_outside_window(self):
        """Trades far apart should not trigger churn."""
        prices = [100.0] * 20
        env, scheme = _make_env(
            rewards.PBR,
            prices=prices,
            commission=0.01,
            churn_window=1,
        )

        env.step(1)  # buy  (step 1)
        env.step(0)  # hold (step 2)
        env.step(0)  # hold (step 3)
        env.step(2)  # sell (step 4, > 1 step from last trade)

        stats = scheme.get_stats()
        assert stats["churn_trade_count"] == 0

    def test_hold_bonus_when_no_trade(self):
        """Hold bonus should be added when not trading and hold_bonus > 0."""
        prices = [100.0] * 10
        env, scheme = _make_env(
            rewards.PBR,
            prices=prices,
            commission=0.0,
            hold_bonus=0.1,
        )

        _, reward, _, _, _ = env.step(0)  # hold
        assert reward == pytest.approx(0.1, abs=1e-6)

    def test_reward_clip(self):
        """Reward clipping should work when reward_clip is set."""
        # Create volatile prices to generate large rewards
        prices = [100.0, 200.0, 50.0, 150.0, 30.0, 300.0, 100.0, 200.0, 50.0, 150.0]
        env, scheme = _make_env(
            rewards.PBR,
            prices=prices,
            commission=0.0,
            reward_clip=0.5,
        )

        env.step(1)  # buy
        # Step through a few volatile prices
        for _ in range(3):
            _, reward, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
            assert -0.5 <= reward <= 0.5 + 1e-9

    def test_reset_clears_all_stats(self):
        """Reset should zero all stats."""
        env, scheme = _make_env(rewards.PBR, commission=0.01)
        env.step(1)
        env.step(2)

        scheme.reset()
        stats = scheme.get_stats()
        assert stats["trade_count"] == 0
        assert stats["cumulative_reward"] == 0.0
        assert stats["total_commission_penalty"] == 0.0
        assert stats["churn_trade_count"] == 0

    def test_cumulative_reward_tracking(self):
        """Cumulative reward should sum all step rewards."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        env, scheme = _make_env(rewards.PBR, prices=prices, commission=0.0)

        total = 0.0
        env.step(1)  # buy at step 0
        for _ in range(4):
            _, reward, terminated, truncated, _ = env.step(0)
            total += reward
            if terminated or truncated:
                break

        stats = scheme.get_stats()
        assert stats["cumulative_reward"] == pytest.approx(total, abs=1e-6)


# ---------------------------------------------------------------------------
# SimpleProfit stats
# ---------------------------------------------------------------------------


class TestSimpleProfitStats:
    """Tests for SimpleProfit.get_stats() and on_action()."""

    def test_on_action_counts(self):
        scheme = rewards.SimpleProfit()
        scheme.on_action(0)
        scheme.on_action(1)
        scheme.on_action(1)
        scheme.on_action(2)
        scheme.on_action(0)

        stats = scheme.get_stats()
        assert stats["buy_count"] == 2
        assert stats["sell_count"] == 1
        assert stats["hold_count"] == 2
        assert stats["trade_count"] == 3

    def test_reset_clears_stats(self):
        scheme = rewards.SimpleProfit()
        scheme.on_action(1)
        scheme.on_action(2)

        scheme.reset()
        stats = scheme.get_stats()
        assert stats["buy_count"] == 0
        assert stats["sell_count"] == 0
        assert stats["hold_count"] == 0
        assert stats["trade_count"] == 0

    def test_get_reward_single_entry(self):
        """Single net worth entry should return 0."""
        portfolio = Portfolio(USD)
        portfolio._performance = OrderedDict({0: {"net_worth": 10000.0}})
        scheme = rewards.SimpleProfit()
        assert scheme.get_reward(portfolio) == 0.0


# ---------------------------------------------------------------------------
# RiskAdjustedReturns stats
# ---------------------------------------------------------------------------


class TestRiskAdjustedReturnsStats:
    """Tests for RiskAdjustedReturns.get_stats() and on_action()."""

    def test_on_action_counts(self):
        scheme = rewards.RiskAdjustedReturns()
        scheme.on_action(1)
        scheme.on_action(2)
        scheme.on_action(0)

        stats = scheme.get_stats()
        assert stats["buy_count"] == 1
        assert stats["sell_count"] == 1
        assert stats["hold_count"] == 1
        assert stats["trade_count"] == 2

    def test_reset_clears_stats(self):
        scheme = rewards.RiskAdjustedReturns()
        scheme.on_action(1)
        scheme.reset()
        stats = scheme.get_stats()
        assert stats == {
            "trade_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }

    def test_invalid_algorithm_raises(self):
        with pytest.raises(AssertionError):
            rewards.RiskAdjustedReturns(return_algorithm="invalid")


# ---------------------------------------------------------------------------
# AdaptiveProfitSeeker
# ---------------------------------------------------------------------------


class TestAdaptiveProfitSeeker:
    """Tests for AdaptiveProfitSeeker reward scheme."""

    def test_init_defaults(self):
        prices = [100.0 + i for i in range(10)]
        price = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.AdaptiveProfitSeeker(price=price)
        assert scheme.participation_bonus == 0.0002
        assert scheme.momentum_weight == 2.0
        assert scheme.drawdown_weight == 0.5
        assert scheme.commission == 0.0001

    def test_custom_params(self):
        prices = [100.0 + i for i in range(10)]
        price = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.AdaptiveProfitSeeker(
            price=price,
            participation_bonus=0.001,
            momentum_weight=3.0,
            drawdown_weight=1.0,
            commission=0.01,
        )
        assert scheme.participation_bonus == 0.001
        assert scheme.momentum_weight == 3.0

    def test_first_reward_is_zero(self):
        """First step should return 0 (initialization)."""
        env, scheme = _make_env(rewards.AdaptiveProfitSeeker)
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_hold_returns_zero_reward(self):
        """Holding cash should yield ~0 reward (no participation bonus)."""
        env, scheme = _make_env(rewards.AdaptiveProfitSeeker, commission=0.0)
        env.step(0)  # first step (init)
        _, reward, _, _, _ = env.step(0)  # second hold
        # With no position, participation_bonus * 0.0 = 0
        assert reward == pytest.approx(0.0, abs=1e-3)

    def test_participation_bonus_when_long(self):
        """Being long should earn participation bonus."""
        prices = [100.0] * 15  # flat prices
        env, scheme = _make_env(
            rewards.AdaptiveProfitSeeker,
            prices=prices,
            commission=0.0,
        )
        env.step(1)  # buy (first step, init → returns 0)
        # After buying, position_frac > 0; participation bonus kicks in
        _, reward, _, _, _ = env.step(0)  # hold while long
        # In a flat market with no commission, reward ≈ participation_bonus * position_frac
        # participation_bonus=0.0002, position_frac should be ~1.0
        assert reward >= 0.0  # at least non-negative

    def test_on_action_stats(self):
        prices = [100.0 + i for i in range(10)]
        price = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.AdaptiveProfitSeeker(price=price)
        scheme.on_action(1)
        scheme.on_action(0)
        scheme.on_action(2)
        stats = scheme.get_stats()
        assert stats["buy_count"] == 1
        assert stats["sell_count"] == 1
        assert stats["hold_count"] == 1
        assert stats["trade_count"] == 2

    def test_reset_clears_state(self):
        prices = [100.0 + i for i in range(10)]
        price = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.AdaptiveProfitSeeker(price=price)
        scheme.on_action(1)
        scheme._prev_net_worth = 10000.0
        scheme._equity_peak = 12000.0

        scheme.reset()
        assert scheme._prev_net_worth == 0.0
        assert scheme._equity_peak == 0.0
        assert scheme.buy_count == 0
        assert scheme.sell_count == 0
        assert scheme.hold_count == 0

    def test_registry(self):
        """AdaptiveProfitSeeker should be in the registry."""
        assert "adaptive-profit-seeker" in rewards._registry
        assert (
            rewards._registry["adaptive-profit-seeker"] == rewards.AdaptiveProfitSeeker
        )


# ---------------------------------------------------------------------------
# Reward registry
# ---------------------------------------------------------------------------


class TestRewardRegistry:
    """Tests for the reward get() registry function."""

    @pytest.mark.parametrize(
        "identifier",
        ["simple", "risk-adjusted", "max-drawdown-penalty"],
    )
    def test_get_returns_instance(self, identifier):
        scheme = rewards.get(identifier)
        assert isinstance(scheme, rewards.TensorTradeRewardScheme)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not associated"):
            rewards.get("nonexistent-scheme")

    def test_all_registry_entries_instantiate(self):
        """Every registered non-price reward should instantiate cleanly."""
        no_price_needed = {"simple", "risk-adjusted", "max-drawdown-penalty"}
        for key in no_price_needed:
            instance = rewards.get(key)
            assert instance is not None
