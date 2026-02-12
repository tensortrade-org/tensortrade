"""Extended tests for reward schemes: PBR stats, AdaptiveProfitSeeker, registry."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
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
        rewards.TrendPBR,
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


# ---------------------------------------------------------------------------
# TrendPBR
# ---------------------------------------------------------------------------


class TestTrendPBR:
    """Tests for TrendPBR trend-aware reward scheme."""

    def test_degrades_to_pbr_with_zero_trend_weight(self):
        """With trend_weight=0, rewards should match PBR exactly."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]

        env_pbr, scheme_pbr = _make_env(rewards.PBR, prices=prices, commission=0.001)
        env_trend, scheme_trend = _make_env(
            rewards.TrendPBR, prices=prices, commission=0.001, trend_weight=0.0
        )

        pbr_rewards = []
        trend_rewards = []

        for action in [1, 0, 0, 0, 2, 0]:
            _, r1, done1, trunc1, _ = env_pbr.step(action)
            _, r2, done2, trunc2, _ = env_trend.step(action)
            if done1 or trunc1 or done2 or trunc2:
                break
            pbr_rewards.append(r1)
            trend_rewards.append(r2)

        for r_pbr, r_trend in zip(pbr_rewards, trend_rewards):
            assert r_trend == pytest.approx(r_pbr, abs=1e-9)

    def test_trend_bonus_positive_in_uptrend_when_long(self):
        """Rising prices + long position should produce positive trend bonus."""
        prices = [100.0 + i * 2.0 for i in range(20)]  # strong uptrend
        env, scheme = _make_env(
            rewards.TrendPBR, prices=prices, commission=0.0, trend_weight=0.01
        )

        env.step(1)  # buy
        # Let EMAs diverge
        rewards_collected = []
        for _ in range(8):
            _, r, done, trunc, _ = env.step(0)
            if done or trunc:
                break
            rewards_collected.append(r)

        # With trend_weight=0.01, trend bonus should be clearly positive
        # Compare to what PBR alone would give (just price diff = ~2.0 each step)
        # Trend adds position * trend_strength * price * trend_weight > 0
        assert sum(rewards_collected) > 0
        stats = scheme.get_stats()
        assert stats["total_trend_reward"] > 0

    def test_trend_bonus_negative_in_downtrend_when_long(self):
        """Falling prices + long position should produce negative trend component."""
        prices = [200.0 - i * 2.0 for i in range(20)]  # strong downtrend
        env, scheme = _make_env(
            rewards.TrendPBR, prices=prices, commission=0.0, trend_weight=0.01
        )

        env.step(1)  # buy
        for _ in range(8):
            _, r, done, trunc, _ = env.step(0)
            if done or trunc:
                break

        stats = scheme.get_stats()
        assert stats["total_trend_reward"] < 0

    def test_trend_bonus_zero_when_cash(self):
        """Position=0 (cash) means no trend effect."""
        prices = [100.0 + i * 5.0 for i in range(20)]
        env, scheme = _make_env(
            rewards.TrendPBR, prices=prices, commission=0.0, trend_weight=0.01
        )

        # Stay in cash (hold)
        for _ in range(8):
            _, r, done, trunc, _ = env.step(0)
            if done or trunc:
                break

        stats = scheme.get_stats()
        assert stats["total_trend_reward"] == pytest.approx(0.0, abs=1e-12)

    def test_trend_strength_zero_for_flat_prices(self):
        """Constant prices → EMAs equal → trend=0."""
        prices = [100.0] * 20
        env, scheme = _make_env(
            rewards.TrendPBR, prices=prices, commission=0.0, trend_weight=0.01
        )

        env.step(1)  # buy
        for _ in range(8):
            _, r, done, trunc, _ = env.step(0)
            if done or trunc:
                break

        stats = scheme.get_stats()
        assert stats["total_trend_reward"] == pytest.approx(0.0, abs=1e-9)

    def test_warmup_sets_ema_state(self):
        """After warmup, EMAs should be non-None and reflect historical data."""
        prices = [100.0 + i for i in range(20)]
        price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.TrendPBR(price=price_stream, trend_weight=0.001)

        assert scheme._fast_ema is None
        assert scheme._slow_ema is None

        scheme.warmup([100.0, 101.0, 102.0, 103.0, 104.0])

        assert scheme._fast_ema is not None
        assert scheme._slow_ema is not None
        # Fast EMA should be closer to recent prices than slow EMA
        assert scheme._fast_ema > scheme._slow_ema

    def test_warmup_produces_immediate_trend_signal(self):
        """Warmup with uptrend data should produce positive trend from step 1."""
        # Generate uptrend warmup data
        warmup_prices = [100.0 + i * 2.0 for i in range(50)]
        prices = [200.0 + i * 2.0 for i in range(20)]

        env, scheme = _make_env(
            rewards.TrendPBR, prices=prices, commission=0.0, trend_weight=0.01
        )

        # Warmup with strong uptrend
        scheme.warmup(warmup_prices)

        env.step(1)  # buy
        _, r, _, _, _ = env.step(0)  # first hold step

        # Trend bonus should be positive immediately (EMAs already diverged)
        stats = scheme.get_stats()
        assert stats["total_trend_reward"] > 0

    def test_reset_clears_warmup(self):
        """After reset, EMAs should be None again."""
        prices = [100.0 + i for i in range(20)]
        price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.TrendPBR(price=price_stream)

        scheme.warmup([100.0, 101.0, 102.0])
        assert scheme._fast_ema is not None

        scheme.reset()
        assert scheme._fast_ema is None
        assert scheme._slow_ema is None

    def test_stats_include_trend_metrics(self):
        """get_stats() should have trend-specific keys."""
        prices = [100.0 + i for i in range(20)]
        env, scheme = _make_env(rewards.TrendPBR, prices=prices, commission=0.0)

        env.step(1)
        env.step(0)

        stats = scheme.get_stats()
        assert "avg_trend_strength" in stats
        assert "uptrend_steps" in stats
        assert "downtrend_steps" in stats
        assert "total_trend_reward" in stats
        # Also has standard PBR stats
        assert "trade_count" in stats
        assert "cumulative_reward" in stats

    def test_churn_penalty_still_works(self):
        """Rapid trading within window should trigger churn penalty."""
        prices = [100.0] * 20
        env, scheme = _make_env(
            rewards.TrendPBR,
            prices=prices,
            commission=0.01,
            churn_window=3,
        )

        env.step(1)  # buy  (step 1)
        env.step(2)  # sell (step 2, within 3 of last)
        env.step(1)  # buy  (step 3, within 3 of step 2)

        stats = scheme.get_stats()
        assert stats["churn_trade_count"] >= 1

    def test_reward_clip_applies_to_total(self):
        """Clip should bound total reward (base + trend - penalty)."""
        prices = [100.0, 200.0, 50.0, 150.0, 30.0, 300.0, 100.0, 200.0, 50.0, 150.0]
        env, scheme = _make_env(
            rewards.TrendPBR,
            prices=prices,
            commission=0.0,
            reward_clip=0.5,
            trend_weight=0.1,  # large trend weight to amplify
        )

        env.step(1)  # buy
        for _ in range(4):
            _, reward, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
            assert -0.5 <= reward <= 0.5 + 1e-9

    def test_registry_entry(self):
        """'trend-pbr' should be in the registry."""
        assert "trend-pbr" in rewards._registry
        assert rewards._registry["trend-pbr"] == rewards.TrendPBR


# ---------------------------------------------------------------------------
# TrendPBR Diagnostics — step-by-step trend signal verification
# ---------------------------------------------------------------------------


@dataclass
class StepSnapshot:
    """Per-step diagnostic snapshot of TrendPBR internals."""

    step: int
    price: float
    reward: float
    fast_ema: float
    slow_ema: float
    trend_strength: float
    trend_component: float
    position: int


def _run_trend_episode(
    prices: list[float],
    actions: list[int],
    commission: float = 0.0,
    trend_weight: float = 0.001,
    trend_scale: float = 20.0,
    warmup_prices: list[float] | None = None,
    **kwargs,
) -> tuple[list[StepSnapshot], rewards.TrendPBR]:
    """Step through a TrendPBR episode, returning per-step diagnostics."""
    env, scheme = _make_env(
        rewards.TrendPBR,
        prices=prices,
        commission=commission,
        trend_weight=trend_weight,
        trend_scale=trend_scale,
        **kwargs,
    )
    if warmup_prices:
        scheme.warmup(warmup_prices)

    snapshots: list[StepSnapshot] = []
    for i, action in enumerate(actions):
        _, reward, done, trunc, _ = env.step(action)
        snapshots.append(
            StepSnapshot(
                step=i,
                price=prices[i + 1] if i + 1 < len(prices) else prices[-1],
                reward=reward,
                fast_ema=scheme._fast_ema or 0.0,
                slow_ema=scheme._slow_ema or 0.0,
                trend_strength=scheme._last_trend_strength,
                trend_component=scheme._last_trend_component,
                position=scheme.position,
            )
        )
        if done or trunc:
            break
    return snapshots, scheme


class TestTrendPBRDiagnostics:
    """Step-by-step trend signal verification.

    These tests simulate what you'd watch in training logs: walking through
    distinct price regimes (uptrend, flat, downtrend, reversal) and checking
    that EMAs, trend_strength, and trend_component behave correctly at each
    phase boundary.
    """

    def test_uptrend_emas_diverge_fast_above_slow(self):
        """In a steady uptrend, fast EMA > slow EMA and the gap widens."""
        prices = [100.0 + i * 1.0 for i in range(40)]
        actions = [1] + [0] * 35  # buy then hold

        snaps, _ = _run_trend_episode(prices, actions)

        # After a few steps for EMAs to warm, fast should lead slow
        for s in snaps[5:]:
            assert s.fast_ema > s.slow_ema, (
                f"step {s.step}: fast_ema {s.fast_ema:.4f} should be > "
                f"slow_ema {s.slow_ema:.4f} in uptrend"
            )

        # Gap should widen over time (fast pulls away from slow)
        early_gap = snaps[8].fast_ema - snaps[8].slow_ema
        late_gap = snaps[25].fast_ema - snaps[25].slow_ema
        assert late_gap > early_gap

    def test_downtrend_emas_diverge_fast_below_slow(self):
        """In a steady downtrend, fast EMA < slow EMA."""
        prices = [200.0 - i * 1.0 for i in range(40)]
        actions = [1] + [0] * 35

        snaps, _ = _run_trend_episode(prices, actions)

        for s in snaps[5:]:
            assert s.fast_ema < s.slow_ema, (
                f"step {s.step}: fast_ema {s.fast_ema:.4f} should be < "
                f"slow_ema {s.slow_ema:.4f} in downtrend"
            )

    def test_trend_strength_positive_in_uptrend(self):
        """trend_strength should be consistently positive while long in uptrend."""
        prices = [100.0 + i * 1.5 for i in range(40)]
        actions = [1] + [0] * 35

        snaps, _ = _run_trend_episode(prices, actions)

        # After warm-in period, every step should have positive trend_strength
        for s in snaps[3:]:
            assert s.trend_strength > 0, (
                f"step {s.step}: trend_strength {s.trend_strength:.6f} "
                f"should be > 0 in uptrend"
            )

    def test_trend_strength_negative_in_downtrend(self):
        """trend_strength should be consistently negative in downtrend."""
        prices = [200.0 - i * 1.5 for i in range(40)]
        actions = [1] + [0] * 35

        snaps, _ = _run_trend_episode(prices, actions)

        for s in snaps[3:]:
            assert s.trend_strength < 0, (
                f"step {s.step}: trend_strength {s.trend_strength:.6f} "
                f"should be < 0 in downtrend"
            )

    def test_trend_strength_grows_then_saturates_in_uptrend(self):
        """In a steady uptrend, trend_strength should grow early then saturate near 1.

        Note: tanh saturates, and the EMA crossover ratio approaches a constant
        in a linear trend, so tiny numerical wobble near saturation is expected.
        We verify: (1) early growth, (2) late saturation above a threshold.
        """
        prices = [100.0 + i * 2.0 for i in range(40)]
        actions = [1] + [0] * 35

        snaps, _ = _run_trend_episode(prices, actions, trend_scale=20.0)

        # Early growth: step 10 should be stronger than step 3
        assert snaps[10].trend_strength > snaps[3].trend_strength, (
            f"trend_strength should grow early: step3={snaps[3].trend_strength:.6f} "
            f"step10={snaps[10].trend_strength:.6f}"
        )

        # Late saturation: should be close to 1.0 (tanh asymptote)
        for s in snaps[20:]:
            assert s.trend_strength > 0.9, (
                f"step {s.step}: trend_strength {s.trend_strength:.6f} "
                f"should saturate above 0.9 in steady uptrend"
            )

    def test_trend_component_zero_when_in_cash(self):
        """Trend component must be exactly 0 for every step in cash."""
        prices = [100.0 + i * 3.0 for i in range(30)]
        actions = [0] * 25  # all hold, stay in cash

        snaps, _ = _run_trend_episode(prices, actions, trend_weight=0.1)

        for s in snaps:
            assert s.trend_component == 0.0, (
                f"step {s.step}: trend_component {s.trend_component} "
                f"should be 0.0 in cash"
            )
            assert s.position == 0

    def test_trend_component_nonzero_when_long_in_trend(self):
        """When long in a trend, trend_component must be nonzero."""
        prices = [100.0 + i * 2.0 for i in range(30)]
        actions = [1] + [0] * 25  # buy then hold

        snaps, _ = _run_trend_episode(prices, actions, trend_weight=0.01)

        # After EMA warm-in, every long-position step should have positive trend_component
        for s in snaps[4:]:
            assert s.trend_component > 0, (
                f"step {s.step}: trend_component {s.trend_component:.8f} "
                f"should be > 0 when long in uptrend"
            )

    def test_reversal_flips_trend_sign(self):
        """Uptrend → downtrend should flip trend_strength from + to -."""
        # 20 steps up, then 30 steps down
        up = [100.0 + i * 2.0 for i in range(20)]
        down = [up[-1] - i * 2.0 for i in range(1, 31)]
        prices = up + down
        actions = [1] + [0] * (len(prices) - 3)  # buy and hold throughout

        snaps, _ = _run_trend_episode(prices, actions)

        # Should be positive during uptrend phase
        uptrend_ts = [s.trend_strength for s in snaps[5:17]]
        assert all(t > 0 for t in uptrend_ts), (
            f"uptrend phase trend_strengths should all be positive: {uptrend_ts}"
        )

        # Should eventually become negative after reversal
        late_ts = [s.trend_strength for s in snaps[-8:]]
        assert all(t < 0 for t in late_ts), (
            f"late downtrend trend_strengths should all be negative: {late_ts}"
        )

    def test_flat_prices_trend_stays_near_zero(self):
        """Constant prices → EMAs equal → trend_strength ≈ 0 at every step."""
        prices = [150.0] * 30
        actions = [1] + [0] * 25

        snaps, _ = _run_trend_episode(prices, actions)

        for s in snaps:
            assert abs(s.trend_strength) < 1e-9, (
                f"step {s.step}: trend_strength {s.trend_strength} "
                f"should be ~0 for flat prices"
            )

    def test_warmup_gives_stronger_initial_signal_than_cold_start(self):
        """Warmup with uptrend data should produce a stronger trend signal
        at step 1 than starting cold."""
        warmup_data = [100.0 + i * 2.0 for i in range(50)]
        prices = [200.0 + i * 2.0 for i in range(20)]
        actions = [1, 0, 0, 0, 0]

        snaps_warm, _ = _run_trend_episode(prices, actions, warmup_prices=warmup_data)
        snaps_cold, _ = _run_trend_episode(prices, actions)

        # After buy + first hold, warmed-up version should have stronger signal
        assert abs(snaps_warm[1].trend_strength) > abs(snaps_cold[1].trend_strength), (
            f"warm trend_strength {snaps_warm[1].trend_strength:.6f} should be "
            f"stronger than cold {snaps_cold[1].trend_strength:.6f}"
        )

    def test_warmup_emas_match_manual_computation(self):
        """Warmup EMAs should match hand-computed EMA values."""
        warmup_data = [100.0, 110.0, 105.0, 115.0, 120.0]
        prices = [125.0] * 20
        price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")
        scheme = rewards.TrendPBR(price=price_stream, ema_fast=3, ema_slow=5)

        # Hand-compute EMAs
        fast_alpha = 2.0 / (3 + 1)  # 0.5
        slow_alpha = 2.0 / (5 + 1)  # 0.333...
        fast_ema = warmup_data[0]
        slow_ema = warmup_data[0]
        for p in warmup_data[1:]:
            fast_ema = fast_alpha * p + (1 - fast_alpha) * fast_ema
            slow_ema = slow_alpha * p + (1 - slow_alpha) * slow_ema

        scheme.warmup(warmup_data)

        assert scheme._fast_ema == pytest.approx(fast_ema, abs=1e-10)
        assert scheme._slow_ema == pytest.approx(slow_ema, abs=1e-10)

    def test_trend_strength_always_bounded(self):
        """trend_strength must stay in [-1, 1] even with extreme price swings."""
        # Wild oscillations
        prices = []
        for i in range(40):
            prices.append(100.0 + ((-1) ** i) * 50.0 * (i + 1))
        actions = [1] + [0] * 35

        snaps, _ = _run_trend_episode(prices, actions, trend_scale=100.0)

        for s in snaps:
            assert -1.0 <= s.trend_strength <= 1.0, (
                f"step {s.step}: trend_strength {s.trend_strength} out of [-1, 1]"
            )

    def test_trend_scale_controls_sensitivity(self):
        """Higher trend_scale should produce stronger trend_strength for same prices."""
        prices = [100.0 + i * 1.0 for i in range(30)]
        actions = [1] + [0] * 25

        snaps_low, _ = _run_trend_episode(prices, actions, trend_scale=5.0)
        snaps_high, _ = _run_trend_episode(prices, actions, trend_scale=50.0)

        # At the same step, higher scale → stronger (or equal, if saturated) signal
        for s_lo, s_hi in zip(snaps_low[5:20], snaps_high[5:20]):
            assert s_hi.trend_strength >= s_lo.trend_strength - 1e-9, (
                f"step {s_lo.step}: scale=50 strength {s_hi.trend_strength:.6f} "
                f"should be >= scale=5 strength {s_lo.trend_strength:.6f}"
            )

    def test_stats_expose_last_trend_strength(self):
        """get_stats() should include last_trend_strength and last_trend_component."""
        prices = [100.0 + i for i in range(20)]
        env, scheme = _make_env(rewards.TrendPBR, prices=prices, commission=0.0)

        env.step(1)
        env.step(0)
        env.step(0)

        stats = scheme.get_stats()
        assert "last_trend_strength" in stats
        assert "last_trend_component" in stats
        # Should reflect the most recent step, not a cumulative
        assert stats["last_trend_strength"] == pytest.approx(
            scheme._last_trend_strength
        )


# ---------------------------------------------------------------------------
# TrendPBR vs Real BTC Data — validate trend signal against known regimes
# ---------------------------------------------------------------------------

# Path to Coinbase daily BTC CSV (shipped with repo in examples/data/)
_BTC_CSV = (
    Path(__file__).resolve().parents[7] / "examples" / "data" / "Coinbase_BTCUSD_d.csv"
)


def _load_btc_daily() -> pd.DataFrame:
    """Load Coinbase daily BTC data, sorted ascending by date."""
    df = pd.read_csv(_BTC_CSV, skiprows=1)
    df.columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume_btc",
        "volume",
    ]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _btc_slice(df: pd.DataFrame, start: str, end: str) -> list[float]:
    """Extract close prices for a date range."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, "close"].tolist()


def _run_trend_on_btc(
    prices: list[float],
    warmup_prices: list[float] | None = None,
) -> tuple[list[StepSnapshot], rewards.TrendPBR]:
    """Run TrendPBR over real BTC prices, long the entire time.

    Pads prices list to be long enough for the env (needs extra for feed).
    Uses trend_weight=0.001 and trend_scale=20 (production defaults).
    """
    # Need enough prices for env setup + steps
    actions = [1] + [0] * (len(prices) - 3)  # buy at start, hold rest
    return _run_trend_episode(
        prices=prices,
        actions=actions,
        commission=0.0,
        trend_weight=0.001,
        trend_scale=20.0,
        warmup_prices=warmup_prices,
    )


@pytest.mark.skipif(not _BTC_CSV.exists(), reason="BTC CSV not found")
class TestTrendPBRvsBTC:
    """Validate TrendPBR trend signal against known BTC market regimes.

    Uses Coinbase daily BTC/USD data to verify that:
    - Bull markets produce positive trend_strength
    - Bear markets produce negative trend_strength
    - Crashes flip the signal negative
    - Warmup with prior data gives correct initial signal
    """

    @pytest.fixture(autouse=True)
    def _load_data(self):
        self.df = _load_btc_daily()

    def test_2017_parabolic_bull_trend_positive(self):
        """Oct-Dec 2017 parabolic rally ($4400 -> $19400): trend should be
        strongly positive through most of the period."""
        prices = _btc_slice(self.df, "2017-10-01", "2017-12-17")
        assert len(prices) >= 40

        snaps, scheme = _run_trend_on_btc(prices)

        # After EMA warm-in (~26 days for slow EMA), trend_strength should be positive
        mid_to_late = [s for s in snaps if s.step >= 30]
        positive_count = sum(1 for s in mid_to_late if s.trend_strength > 0)
        total = len(mid_to_late)
        positive_pct = positive_count / total if total > 0 else 0

        assert positive_pct > 0.9, (
            f"2017 bull: only {positive_pct:.0%} of steps had positive trend "
            f"(expected >90%). Strengths: "
            f"{[f'{s.trend_strength:.3f}' for s in mid_to_late[:10]]}"
        )

        # Trend should be strong (near saturation) in late parabolic phase
        late_strengths = [s.trend_strength for s in snaps[-10:]]
        assert all(t > 0.5 for t in late_strengths), (
            f"Late 2017 parabolic trend should be >0.5: {late_strengths}"
        )

    def test_jan_2018_crash_trend_flips_negative(self):
        """Jan 2018 crash ($17k -> $7.7k): trend should flip from positive to
        negative during the crash."""
        # Include Dec 2017 as warmup context
        warmup = _btc_slice(self.df, "2017-11-01", "2017-12-31")
        crash_prices = _btc_slice(self.df, "2018-01-01", "2018-02-06")
        assert len(crash_prices) >= 25

        snaps, scheme = _run_trend_on_btc(crash_prices, warmup_prices=warmup)

        # Early: should still have positive trend from the prior bull warmup
        early_strength = snaps[2].trend_strength
        assert early_strength > 0, (
            f"Early Jan 2018 should still have positive trend from bull warmup: "
            f"{early_strength:.4f}"
        )

        # Late: should have flipped negative from the crash
        late_strengths = [s.trend_strength for s in snaps[-5:]]
        assert all(t < 0 for t in late_strengths), (
            f"Late Jan 2018 crash should have negative trend: {late_strengths}"
        )

    def test_nov_2018_crash_trend_negative(self):
        """Nov 2018 crash ($6300 -> $3900): after warm-in, trend should be
        consistently negative."""
        # Use Oct 2018 as warmup (sideways ~$6300-$6500)
        warmup = _btc_slice(self.df, "2018-10-01", "2018-10-31")
        crash_prices = _btc_slice(self.df, "2018-11-01", "2018-11-30")
        assert len(crash_prices) >= 20

        snaps, scheme = _run_trend_on_btc(crash_prices, warmup_prices=warmup)

        # Mid-to-late November: should be negative
        mid_snaps = [s for s in snaps if s.step >= 10]
        negative_count = sum(1 for s in mid_snaps if s.trend_strength < 0)
        total = len(mid_snaps)
        negative_pct = negative_count / total if total > 0 else 0

        assert negative_pct > 0.8, (
            f"Nov 2018 crash: only {negative_pct:.0%} negative (expected >80%)"
        )

    def test_q2_2019_rally_trend_positive(self):
        """Apr-Jun 2019 rally ($4100 -> $12900): trend should be positive
        through most of the rally."""
        prices = _btc_slice(self.df, "2019-04-01", "2019-06-26")
        assert len(prices) >= 50

        snaps, scheme = _run_trend_on_btc(prices)

        # After warm-in, should be predominantly positive
        warmed = [s for s in snaps if s.step >= 26]
        positive_count = sum(1 for s in warmed if s.trend_strength > 0)
        total = len(warmed)
        positive_pct = positive_count / total if total > 0 else 0

        assert positive_pct > 0.85, (
            f"Q2 2019 rally: only {positive_pct:.0%} positive (expected >85%)"
        )

    def test_2018_bear_market_predominantly_negative(self):
        """Full year 2018 bear ($13500 -> $3700): after initial warm-in,
        trend should be predominantly negative."""
        prices = _btc_slice(self.df, "2018-01-01", "2018-12-31")
        assert len(prices) >= 300

        snaps, scheme = _run_trend_on_btc(prices)

        # Skip first 50 days (Jan still had residual bull momentum)
        bear_snaps = [s for s in snaps if s.step >= 50]
        negative_count = sum(1 for s in bear_snaps if s.trend_strength < 0)
        total = len(bear_snaps)
        negative_pct = negative_count / total if total > 0 else 0

        assert negative_pct > 0.6, (
            f"2018 bear: only {negative_pct:.0%} negative (expected >60%). "
            f"Bear markets have relief rallies so 60% is the floor."
        )

    def test_warmup_with_bull_gives_correct_initial_signal(self):
        """Warming up with 2017 bull data should give positive signal from
        step 1 of 2018 Jan crash data."""
        warmup = _btc_slice(self.df, "2017-09-01", "2017-12-31")
        prices = _btc_slice(self.df, "2018-01-01", "2018-01-31")

        snaps, scheme = _run_trend_on_btc(prices, warmup_prices=warmup)

        # Step 0 (buy) and step 1 should have positive trend from bull warmup
        assert snaps[0].trend_strength > 0, (
            f"After bull warmup, initial trend should be positive: "
            f"{snaps[0].trend_strength:.4f}"
        )

    def test_warmup_with_bear_gives_correct_initial_signal(self):
        """Warming up with mid-2018 bear data should give negative signal."""
        warmup = _btc_slice(self.df, "2018-06-01", "2018-10-31")
        prices = _btc_slice(self.df, "2018-11-01", "2018-11-30")

        snaps, scheme = _run_trend_on_btc(prices, warmup_prices=warmup)

        assert snaps[0].trend_strength < 0, (
            f"After bear warmup, initial trend should be negative: "
            f"{snaps[0].trend_strength:.4f}"
        )

    def test_trend_strength_stays_bounded_on_real_data(self):
        """trend_strength must stay in [-1, 1] across all BTC history."""
        # Use full dataset
        prices = self.df["close"].tolist()
        snaps, _ = _run_trend_on_btc(prices)

        for s in snaps:
            assert -1.0 <= s.trend_strength <= 1.0, (
                f"step {s.step}: trend_strength {s.trend_strength} out of bounds"
            )
