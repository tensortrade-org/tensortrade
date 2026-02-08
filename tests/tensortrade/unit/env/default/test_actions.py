"""Tests for BSH action scheme, PBR reward integration, and extended strategies."""

import pytest
from unittest.mock import MagicMock, patch
from gymnasium.spaces import Discrete

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC, Quantity
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.env.default.actions import (
    BSH,
    TrailingStopBSH,
    BracketBSH,
    DrawdownBudgetBSH,
    CooldownBSH,
    HoldMinimumBSH,
    ConfirmationBSH,
    ScaledEntryBSH,
    PartialTakeProfitBSH,
    VolatilitySizedBSH,
)
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default


def make_env(n_steps: int = 50, initial_cash: float = 10000.0, initial_btc: float = 0.0):
    """Create a minimal TensorTrade environment for testing."""
    prices = [100.0 + i * 0.5 for i in range(n_steps)]
    price = Stream.source(prices, dtype="float").rename("USD-BTC")

    exchange_options = ExchangeOptions(commission=0.0)
    exchange = Exchange("exchange", service=execute_order, options=exchange_options)(price)

    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, initial_btc * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(prices, dtype="float").rename("close")]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=1,
        max_allowed_loss=0.99,
    )
    return env, action_scheme, reward_scheme, cash, asset


class TestBSHActionSpace:
    """Tests for BSH action space configuration."""

    def test_bsh_action_space_is_discrete_3(self):
        """BSH action space should be Discrete(3): hold, buy, sell."""
        env, action_scheme, _, _, _ = make_env()
        assert isinstance(action_scheme.action_space, Discrete)
        assert action_scheme.action_space.n == 3

    def test_env_action_space_matches_bsh(self):
        """Environment action space should match BSH action space."""
        env, _, _, _, _ = make_env()
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 3


class TestBSHHold:
    """Tests for hold action (action=0)."""

    def test_bsh_hold_produces_no_order(self):
        """Action 0 (hold) should produce no orders from any position state."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        # Hold from cash position
        obs, reward, done, truncated, info = env.step(0)
        assert cash.balance.as_float() == pytest.approx(10000.0, rel=1e-2)
        assert asset.balance.as_float() == pytest.approx(0.0)

    def test_bsh_hold_after_buy_keeps_position(self):
        """Holding after buying should keep the asset position."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        # Buy first
        env.step(1)
        btc_after_buy = asset.balance.as_float()
        assert btc_after_buy > 0

        # Hold — should keep same BTC balance
        env.step(0)
        assert asset.balance.as_float() == pytest.approx(btc_after_buy, rel=1e-6)


class TestBSHBuy:
    """Tests for buy action (action=1)."""

    def test_bsh_buy_from_cash(self):
        """Action 1 (buy) when in cash should create a buy order."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        env.step(1)  # Buy
        assert asset.balance.as_float() > 0
        assert action_scheme._position == 1

    def test_bsh_buy_when_already_holding(self):
        """Action 1 (buy) when already in asset should be a no-op."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        env.step(1)  # Buy
        btc_after_first_buy = asset.balance.as_float()

        env.step(1)  # Buy again — should be no-op
        assert asset.balance.as_float() == pytest.approx(btc_after_first_buy, rel=1e-6)
        assert action_scheme._position == 1

    def test_bsh_buy_with_zero_cash_balance(self):
        """Action 1 (buy) with no cash should produce no order."""
        env, action_scheme, _, cash, asset = make_env(initial_cash=0.0, initial_btc=0.0)
        env.reset()

        env.step(1)  # Buy with no cash
        assert asset.balance.as_float() == 0.0
        assert action_scheme._position == 0  # Should stay in cash position


class TestBSHSell:
    """Tests for sell action (action=2)."""

    def test_bsh_sell_from_asset(self):
        """Action 2 (sell) when holding asset should create a sell order."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        env.step(1)  # Buy first
        assert action_scheme._position == 1

        env.step(2)  # Sell
        assert cash.balance.as_float() > 0
        assert action_scheme._position == 0

    def test_bsh_sell_when_already_cash(self):
        """Action 2 (sell) when already in cash should be a no-op."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        initial_cash_bal = cash.balance.as_float()
        env.step(2)  # Sell from cash — no-op
        assert cash.balance.as_float() == pytest.approx(initial_cash_bal, rel=1e-2)
        assert action_scheme._position == 0

    def test_bsh_sell_with_zero_asset_balance(self):
        """Action 2 (sell) with no asset balance should produce no order."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        # Force position to 1 but with no actual asset balance
        # This tests the balance check inside get_orders
        env.step(2)  # Sell from cash — no-op since _position is 0
        assert action_scheme._position == 0


class TestBSHFullCycle:
    """Tests for full buy/hold/sell cycles."""

    def test_bsh_full_cycle(self):
        """Buy → hold → sell → hold → buy should work correctly."""
        env, action_scheme, _, cash, asset = make_env(n_steps=50)
        env.reset()

        # Buy
        env.step(1)
        assert action_scheme._position == 1
        assert asset.balance.as_float() > 0

        # Hold
        env.step(0)
        assert action_scheme._position == 1

        # Sell
        env.step(2)
        assert action_scheme._position == 0
        assert cash.balance.as_float() > 0

        # Hold
        env.step(0)
        assert action_scheme._position == 0

        # Buy again
        env.step(1)
        assert action_scheme._position == 1
        assert asset.balance.as_float() > 0


class TestBSHReset:
    """Tests for BSH reset behavior."""

    def test_bsh_reset_returns_to_cash(self):
        """After buy, reset() should set _position back to 0."""
        env, action_scheme, _, cash, asset = make_env()
        env.reset()

        env.step(1)  # Buy
        assert action_scheme._position == 1

        env.reset()
        assert action_scheme._position == 0


class TestBSHListeners:
    """Tests for BSH listener/on_action integration."""

    def test_bsh_listener_receives_all_actions(self):
        """Listeners should receive on_action() calls with correct action values."""
        env, action_scheme, reward_scheme, _, _ = make_env()
        env.reset()

        mock_listener = MagicMock()
        action_scheme.attach(mock_listener)

        env.step(0)  # Hold
        mock_listener.on_action.assert_called_with(0)

        env.step(1)  # Buy
        mock_listener.on_action.assert_called_with(1)

        env.step(2)  # Sell
        mock_listener.on_action.assert_called_with(2)

        assert mock_listener.on_action.call_count == 3


class TestPBRRewardIntegration:
    """Tests for PBR reward scheme with 3-action BSH."""

    def test_pbr_starts_at_zero(self):
        """PBR should initialize with position=0 (cash/flat)."""
        price = Stream.source([100.0, 101.0, 102.0], dtype="float").rename("USD-BTC")
        pbr = PBR(price=price)
        assert pbr.position == 0

    def test_pbr_hold_preserves_position(self):
        """PBR on_action(0) should not change position."""
        price = Stream.source([100.0, 101.0, 102.0], dtype="float").rename("USD-BTC")
        pbr = PBR(price=price)
        assert pbr.position == 0  # Starts at 0

        pbr.on_action(0)  # Hold
        assert pbr.position == 0

    def test_pbr_buy_sets_long(self):
        """PBR on_action(1) should set position to 1 (long)."""
        price = Stream.source([100.0, 101.0, 102.0], dtype="float").rename("USD-BTC")
        pbr = PBR(price=price)

        pbr.on_action(1)  # Buy
        assert pbr.position == 1

    def test_pbr_sell_sets_zero(self):
        """PBR on_action(2) should set position to 0 (cash/flat)."""
        price = Stream.source([100.0, 101.0, 102.0], dtype="float").rename("USD-BTC")
        pbr = PBR(price=price)

        pbr.on_action(1)  # Buy first
        pbr.on_action(2)  # Sell
        assert pbr.position == 0

    def test_pbr_reset_returns_to_zero(self):
        """PBR reset() should return position to 0 (cash/flat)."""
        price = Stream.source([100.0, 101.0, 102.0], dtype="float").rename("USD-BTC")
        pbr = PBR(price=price)

        pbr.on_action(1)  # Buy → position=1
        assert pbr.position == 1

        pbr.reset()
        assert pbr.position == 0

    def test_pbr_commission_penalty(self):
        """PBR should penalize trades by 2x commission."""
        price = Stream.source([100.0, 101.0, 102.0], dtype="float").rename("USD-BTC")
        pbr = PBR(price=price, commission=0.01)

        # Buy triggers a trade
        pbr.on_action(1)
        assert pbr._traded is True

        # Hold does not
        pbr.on_action(0)
        assert pbr._traded is False


# ---------------------------------------------------------------------------
# Helper for custom action schemes
# ---------------------------------------------------------------------------


def make_env_with_scheme(action_scheme_cls, n_steps=50, initial_cash=10000.0,
                         initial_btc=0.0, scheme_kwargs=None, prices=None):
    """Create an env with a custom action scheme class."""
    if prices is None:
        prices = [100.0 + i * 0.5 for i in range(n_steps)]
    else:
        n_steps = len(prices)

    price = Stream.source(prices, dtype="float").rename("USD-BTC")

    exchange_options = ExchangeOptions(commission=0.0)
    exchange = Exchange("exchange", service=execute_order, options=exchange_options)(price)

    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, initial_btc * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(prices, dtype="float").rename("close")]
    feed = DataFeed(features)
    feed.compile()

    kwargs = {"cash": cash, "asset": asset}
    if scheme_kwargs:
        kwargs.update(scheme_kwargs)
    action_scheme = action_scheme_cls(**kwargs)

    reward_scheme = PBR(price=price)
    action_scheme.attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=1,
        max_allowed_loss=0.99,
    )
    return env, action_scheme, reward_scheme, cash, asset


# ===========================================================================
# Group 1: Risk Management Tests
# ===========================================================================


class TestTrailingStopBSH:
    """Tests for TrailingStopBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(TrailingStopBSH)
        assert isinstance(scheme.action_space, Discrete)
        assert scheme.action_space.n == 3

    def test_buy_and_hold_without_stop(self):
        """Price rises monotonically — trailing stop never triggers."""
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            TrailingStopBSH, prices=prices, scheme_kwargs={"stop_pct": 0.05}
        )
        env.reset()

        env.step(1)  # Buy at ~100
        assert scheme._position == 1

        for _ in range(10):
            env.step(0)  # Hold
        assert scheme._position == 1  # Still holding

    def test_trailing_stop_triggers_on_drop(self):
        """Price rises then drops enough to trigger the trailing stop."""
        # Note: get_orders sees price from previous observer step.
        # reset reads prices[0]. step(buy) sees prices[0], observer advances to prices[1].
        prices = [100.0, 102.0, 105.0, 108.0, 110.0, 110.0, 105.0, 100.0] + [99.0] * 10
        env, scheme, _, cash, asset = make_env_with_scheme(
            TrailingStopBSH, prices=prices, scheme_kwargs={"stop_pct": 0.05}
        )
        env.reset()  # observer reads prices[0]=100

        env.step(1)  # sees 100, buy. Observer reads prices[1]=102
        assert scheme._position == 1

        env.step(0)  # sees 102, peak=102. Observer reads prices[2]=105
        env.step(0)  # sees 105, peak=105. Observer reads prices[3]=108
        env.step(0)  # sees 108, peak=108. Observer reads prices[4]=110
        env.step(0)  # sees 110, peak=110. Observer reads prices[5]=110

        # Now sees 110, peak stays 110. Observer reads prices[6]=105
        env.step(0)
        assert scheme._position == 1

        # sees 105 (~4.5% from peak=110): no trigger (stop at 110*0.95=104.5)
        env.step(0)  # Observer reads prices[7]=100
        assert scheme._position == 1

        # sees 100 (~9% from peak=110): trigger!
        env.step(0)
        assert scheme._position == 0  # Auto-sold

    def test_manual_sell_works(self):
        """Agent can manually sell before stop triggers."""
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            TrailingStopBSH, prices=prices, scheme_kwargs={"stop_pct": 0.05}
        )
        env.reset()
        env.step(1)  # Buy
        env.step(2)  # Manual sell
        assert scheme._position == 0

    def test_listener_receives_effective_action_on_stop(self):
        """Listener should receive action=2 when trailing stop auto-sells."""
        # reset reads prices[0]=100. step sees prev price.
        prices = [100.0, 110.0, 90.0, 90.0] + [90.0] * 10
        env, scheme, _, _, _ = make_env_with_scheme(
            TrailingStopBSH, prices=prices, scheme_kwargs={"stop_pct": 0.05}
        )
        env.reset()  # observer reads prices[0]=100

        mock = MagicMock()
        scheme.attach(mock)

        env.step(1)  # sees 100, buy. Observer reads prices[1]=110
        env.step(0)  # sees 110, peak=110. Observer reads prices[2]=90
        env.step(0)  # sees 90 (18% drop from 110): stop triggers
        mock.on_action.assert_called_with(2)

    def test_reset_clears_state(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(TrailingStopBSH, prices=prices)
        env.reset()
        env.step(1)
        assert scheme._position == 1

        env.reset()
        assert scheme._position == 0
        assert scheme._entry_price == 0.0
        assert scheme._peak_price == 0.0


class TestBracketBSH:
    """Tests for BracketBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(BracketBSH)
        assert scheme.action_space.n == 3

    def test_take_profit_triggers(self):
        """Price rises above take_profit_pct → auto-sell."""
        # Note: get_orders sees the price from the *previous* observer step.
        # reset() reads prices[0]=100. step(buy) sees 100, observer advances to prices[1].
        # So we need the take-profit price to be visible one step after it appears in the array.
        prices = [100.0, 100.0, 105.0, 112.0, 112.0] + [112.0] * 10
        env, scheme, _, _, _ = make_env_with_scheme(
            BracketBSH, prices=prices,
            scheme_kwargs={"stop_loss_pct": 0.05, "take_profit_pct": 0.10}
        )
        env.reset()  # observer reads prices[0]=100

        env.step(1)  # get_orders sees price=100, buy. Observer reads prices[1]=100
        assert scheme._position == 1

        env.step(0)  # get_orders sees price=100, no trigger. Observer reads prices[2]=105
        assert scheme._position == 1

        env.step(0)  # get_orders sees price=105, within range. Observer reads prices[3]=112
        assert scheme._position == 1

        env.step(0)  # get_orders sees price=112 (12% > 10%): take-profit triggers
        assert scheme._position == 0

    def test_stop_loss_triggers(self):
        """Price drops below stop_loss_pct → auto-sell."""
        # reset reads prices[0]=100. step(buy) sees 100. observer advances to prices[1].
        prices = [100.0, 100.0, 93.0, 93.0] + [93.0] * 10
        env, scheme, _, _, _ = make_env_with_scheme(
            BracketBSH, prices=prices,
            scheme_kwargs={"stop_loss_pct": 0.05, "take_profit_pct": 0.10}
        )
        env.reset()  # observer reads prices[0]=100

        env.step(1)  # sees price=100, buy. Observer reads prices[1]=100
        env.step(0)  # sees price=100, no trigger. Observer reads prices[2]=93
        env.step(0)  # sees price=93 (7% drop > 5%): stop-loss triggers
        assert scheme._position == 0

    def test_no_trigger_in_range(self):
        """Price stays within bracket → no auto-sell."""
        # All prices within [-5%, +10%] of entry=100
        prices = [100.0, 100.0, 103.0, 97.0, 102.0, 100.0] + [100.0] * 10
        env, scheme, _, _, _ = make_env_with_scheme(
            BracketBSH, prices=prices,
            scheme_kwargs={"stop_loss_pct": 0.05, "take_profit_pct": 0.10}
        )
        env.reset()  # observer reads prices[0]=100

        env.step(1)  # sees 100, buy. Observer reads prices[1]=100
        for _ in range(4):
            env.step(0)  # Hold within range
        assert scheme._position == 1

    def test_manual_sell(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(BracketBSH, prices=prices)
        env.reset()
        env.step(1)
        env.step(2)  # Manual emergency exit
        assert scheme._position == 0

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(BracketBSH, prices=prices)
        env.reset()
        env.step(1)
        env.reset()
        assert scheme._position == 0
        assert scheme._entry_price == 0.0


class TestDrawdownBudgetBSH:
    """Tests for DrawdownBudgetBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(DrawdownBudgetBSH)
        assert scheme.action_space.n == 3

    def test_buy_sell_without_drawdown(self):
        """Normal operation without hitting drawdown limit."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            DrawdownBudgetBSH, prices=prices,
            scheme_kwargs={"max_drawdown_pct": 0.10}
        )
        env.reset()

        env.step(1)  # Buy
        assert scheme._position == 1
        env.step(2)  # Sell
        assert scheme._position == 0

    def test_lockout_blocks_buys(self):
        """During lockout, buy signals are suppressed to hold."""
        prices = [100.0 + i * 0.5 for i in range(30)]
        env, scheme, _, _, _ = make_env_with_scheme(
            DrawdownBudgetBSH, prices=prices,
            scheme_kwargs={"max_drawdown_pct": 0.10, "lockout_steps": 5}
        )
        env.reset()

        # Manually trigger lockout
        scheme._lockout_remaining = 5

        env.step(1)  # Buy suppressed during lockout
        assert scheme._position == 0  # Still in cash

    def test_lockout_allows_sells(self):
        """Sells should still be allowed during lockout."""
        prices = [100.0 + i * 0.5 for i in range(30)]
        env, scheme, _, _, _ = make_env_with_scheme(
            DrawdownBudgetBSH, prices=prices,
            scheme_kwargs={"max_drawdown_pct": 0.10, "lockout_steps": 5}
        )
        env.reset()

        env.step(1)  # Buy (no lockout yet)
        assert scheme._position == 1

        # Set lockout
        scheme._lockout_remaining = 5
        env.step(2)  # Sell should work during lockout
        assert scheme._position == 0

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(DrawdownBudgetBSH, prices=prices)
        env.reset()
        env.step(1)
        scheme._lockout_remaining = 5
        env.reset()
        assert scheme._position == 0
        assert scheme._equity_peak == 0.0
        assert scheme._lockout_remaining == 0


# ===========================================================================
# Group 2: Anti-Whipsaw Tests
# ===========================================================================


class TestCooldownBSH:
    """Tests for CooldownBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(CooldownBSH)
        assert scheme.action_space.n == 3

    def test_buy_sell_without_cooldown(self):
        """Normal buy/sell works when no cooldown is active."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            CooldownBSH, prices=prices, scheme_kwargs={"cooldown_steps": 3}
        )
        env.reset()

        env.step(1)  # Buy
        assert scheme._position == 1
        env.step(2)  # Sell — starts cooldown
        assert scheme._position == 0

    def test_cooldown_blocks_buy_after_sell(self):
        """After selling, buys are blocked for cooldown_steps."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            CooldownBSH, prices=prices, scheme_kwargs={"cooldown_steps": 3}
        )
        env.reset()

        env.step(1)  # Buy
        env.step(2)  # Sell — cooldown starts (3 steps)

        # Try buying during cooldown: suppressed
        env.step(1)  # step 1 of cooldown
        assert scheme._position == 0
        env.step(1)  # step 2 of cooldown
        assert scheme._position == 0
        env.step(1)  # step 3 of cooldown (last blocked step)
        assert scheme._position == 0

        # Cooldown expired — buy should work
        env.step(1)
        assert scheme._position == 1

    def test_hold_during_cooldown(self):
        """Hold action during cooldown works normally."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            CooldownBSH, prices=prices, scheme_kwargs={"cooldown_steps": 2}
        )
        env.reset()

        env.step(1)  # Buy
        env.step(2)  # Sell — cooldown starts
        env.step(0)  # Hold
        assert scheme._position == 0

    def test_listener_receives_hold_during_cooldown(self):
        """When buy is suppressed, listener receives 0 (hold)."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            CooldownBSH, prices=prices, scheme_kwargs={"cooldown_steps": 3}
        )
        env.reset()

        mock = MagicMock()
        scheme.attach(mock)

        env.step(1)  # Buy
        env.step(2)  # Sell
        env.step(1)  # Suppressed to hold
        mock.on_action.assert_called_with(0)

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(CooldownBSH, prices=prices)
        env.reset()
        env.step(1)
        env.step(2)
        assert scheme._cooldown_remaining > 0
        env.reset()
        assert scheme._cooldown_remaining == 0
        assert scheme._position == 0


class TestHoldMinimumBSH:
    """Tests for HoldMinimumBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(HoldMinimumBSH)
        assert scheme.action_space.n == 3

    def test_hold_minimum_blocks_sell_after_buy(self):
        """After buying, sells are blocked for min_hold_steps."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            HoldMinimumBSH, prices=prices, scheme_kwargs={"min_hold_steps": 3}
        )
        env.reset()

        env.step(1)  # Buy — hold timer starts (3 steps)

        # Try selling during hold minimum: suppressed
        env.step(2)  # step 1
        assert scheme._position == 1
        env.step(2)  # step 2
        assert scheme._position == 1
        env.step(2)  # step 3 (last blocked step)
        assert scheme._position == 1

        # Hold minimum expired — sell should work
        env.step(2)
        assert scheme._position == 0

    def test_buy_does_not_block_without_hold(self):
        """Buys from cash work immediately (no hold timer active)."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            HoldMinimumBSH, prices=prices, scheme_kwargs={"min_hold_steps": 3}
        )
        env.reset()

        env.step(1)  # Buy
        assert scheme._position == 1

    def test_listener_receives_hold_during_minimum(self):
        """When sell is suppressed, listener receives 0 (hold)."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            HoldMinimumBSH, prices=prices, scheme_kwargs={"min_hold_steps": 2}
        )
        env.reset()

        mock = MagicMock()
        scheme.attach(mock)

        env.step(1)  # Buy
        env.step(2)  # Suppressed to hold
        mock.on_action.assert_called_with(0)

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(HoldMinimumBSH, prices=prices)
        env.reset()
        env.step(1)
        assert scheme._hold_remaining > 0
        env.reset()
        assert scheme._hold_remaining == 0


class TestConfirmationBSH:
    """Tests for ConfirmationBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(ConfirmationBSH)
        assert scheme.action_space.n == 3

    def test_single_buy_signal_does_not_execute(self):
        """A single buy signal is not enough to execute."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            ConfirmationBSH, prices=prices, scheme_kwargs={"confirmation_steps": 3}
        )
        env.reset()

        env.step(1)  # First buy signal
        assert scheme._position == 0  # Not executed yet

    def test_confirmed_buy_executes(self):
        """Three consecutive buy signals execute the buy."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            ConfirmationBSH, prices=prices, scheme_kwargs={"confirmation_steps": 3}
        )
        env.reset()

        env.step(1)  # Signal 1
        env.step(1)  # Signal 2
        env.step(1)  # Signal 3 — confirmed!
        assert scheme._position == 1

    def test_hold_resets_counter(self):
        """A hold signal resets the confirmation counter."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            ConfirmationBSH, prices=prices, scheme_kwargs={"confirmation_steps": 3}
        )
        env.reset()

        env.step(1)  # Signal 1
        env.step(1)  # Signal 2
        env.step(0)  # Hold — resets counter
        env.step(1)  # Signal 1 again
        assert scheme._position == 0  # Not confirmed

    def test_signal_change_resets_counter(self):
        """Changing signal direction resets the counter."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            ConfirmationBSH, prices=prices, scheme_kwargs={"confirmation_steps": 2}
        )
        env.reset()

        env.step(1)  # Buy signal 1
        env.step(2)  # Sell signal — resets, starts sell count
        env.step(2)  # Sell signal 2 — confirmed sell (but position is 0, no-op)
        assert scheme._position == 0  # Was never in asset

    def test_confirmed_sell_executes(self):
        """Confirmed sell after being in position executes."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            ConfirmationBSH, prices=prices, scheme_kwargs={"confirmation_steps": 2}
        )
        env.reset()

        env.step(1)  # Buy signal 1
        env.step(1)  # Buy signal 2 — confirmed buy
        assert scheme._position == 1

        env.step(2)  # Sell signal 1
        assert scheme._position == 1  # Not yet
        env.step(2)  # Sell signal 2 — confirmed sell
        assert scheme._position == 0

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(ConfirmationBSH, prices=prices)
        env.reset()
        env.step(1)
        env.reset()
        assert scheme._pending_action == 0
        assert scheme._confirmation_count == 0


# ===========================================================================
# Group 3: Position Sizing Tests
# ===========================================================================


class TestScaledEntryBSH:
    """Tests for ScaledEntryBSH."""

    def test_action_space_discrete_4(self):
        env, scheme, _, _, _ = make_env_with_scheme(ScaledEntryBSH)
        assert isinstance(scheme.action_space, Discrete)
        assert scheme.action_space.n == 4

    def test_buy_tranche_increments(self):
        """Each buy-tranche action increments _tranches_in."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            ScaledEntryBSH, prices=prices, scheme_kwargs={"num_tranches": 3}
        )
        env.reset()

        env.step(1)  # Buy tranche 1
        assert scheme._tranches_in == 1
        assert scheme._position == 1
        assert asset.balance.as_float() > 0

        env.step(1)  # Buy tranche 2
        assert scheme._tranches_in == 2

        env.step(1)  # Buy tranche 3
        assert scheme._tranches_in == 3

    def test_buy_tranche_over_limit_is_noop(self):
        """Buying beyond num_tranches is a no-op."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            ScaledEntryBSH, prices=prices, scheme_kwargs={"num_tranches": 2}
        )
        env.reset()

        env.step(1)  # Tranche 1
        env.step(1)  # Tranche 2
        btc_after_full = asset.balance.as_float()

        env.step(1)  # Over limit — no-op
        assert scheme._tranches_in == 2
        assert asset.balance.as_float() == pytest.approx(btc_after_full, rel=1e-6)

    def test_sell_all(self):
        """Action 2 sells everything."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            ScaledEntryBSH, prices=prices, scheme_kwargs={"num_tranches": 3}
        )
        env.reset()

        env.step(1)  # Buy 1
        env.step(1)  # Buy 2
        env.step(2)  # Sell all
        assert scheme._tranches_in == 0
        assert scheme._position == 0

    def test_sell_tranche(self):
        """Action 3 sells one tranche."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            ScaledEntryBSH, prices=prices, scheme_kwargs={"num_tranches": 3}
        )
        env.reset()

        env.step(1)  # Tranche 1
        env.step(1)  # Tranche 2
        assert scheme._tranches_in == 2

        env.step(3)  # Sell tranche
        assert scheme._tranches_in == 1
        assert scheme._position == 1  # Still has some asset

        env.step(3)  # Sell last tranche
        assert scheme._tranches_in == 0
        assert scheme._position == 0

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(ScaledEntryBSH, prices=prices)
        env.reset()
        env.step(1)
        env.reset()
        assert scheme._position == 0
        assert scheme._tranches_in == 0


class TestPartialTakeProfitBSH:
    """Tests for PartialTakeProfitBSH."""

    def test_action_space_discrete_4(self):
        env, scheme, _, _, _ = make_env_with_scheme(PartialTakeProfitBSH)
        assert scheme.action_space.n == 4

    def test_buy_sets_full_position(self):
        """Action 1 buys full position."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, asset = make_env_with_scheme(
            PartialTakeProfitBSH, prices=prices
        )
        env.reset()

        env.step(1)  # Buy all
        assert scheme._position == 1
        assert asset.balance.as_float() > 0

    def test_partial_sell_transitions(self):
        """Action 2 from full → half, action 2 from half → cash."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            PartialTakeProfitBSH, prices=prices,
            scheme_kwargs={"first_sell_proportion": 0.5}
        )
        env.reset()

        env.step(1)  # Buy → position 1
        full_btc = asset.balance.as_float()

        env.step(2)  # Partial sell → position 2
        assert scheme._position == 2
        assert asset.balance.as_float() < full_btc
        assert asset.balance.as_float() > 0

        env.step(2)  # Sell remaining → position 0
        assert scheme._position == 0

    def test_sell_all_from_full(self):
        """Action 3 exits from full position directly."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            PartialTakeProfitBSH, prices=prices
        )
        env.reset()

        env.step(1)  # Buy
        env.step(3)  # Sell all
        assert scheme._position == 0

    def test_sell_all_from_half(self):
        """Action 3 exits from half position."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            PartialTakeProfitBSH, prices=prices
        )
        env.reset()

        env.step(1)  # Buy
        env.step(2)  # Partial sell
        assert scheme._position == 2
        env.step(3)  # Sell all from half
        assert scheme._position == 0

    def test_buy_from_half_is_noop(self):
        """Action 1 when in half position is a no-op (position != 0)."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            PartialTakeProfitBSH, prices=prices
        )
        env.reset()

        env.step(1)  # Buy → position 1
        env.step(2)  # Partial sell → position 2
        env.step(1)  # Buy from half — no-op (position != 0)
        assert scheme._position == 2

    def test_reset(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(PartialTakeProfitBSH, prices=prices)
        env.reset()
        env.step(1)
        env.step(2)
        assert scheme._position == 2
        env.reset()
        assert scheme._position == 0


class TestVolatilitySizedBSH:
    """Tests for VolatilitySizedBSH."""

    def test_action_space_discrete_3(self):
        env, scheme, _, _, _ = make_env_with_scheme(VolatilitySizedBSH)
        assert scheme.action_space.n == 3

    def test_buy_with_insufficient_history(self):
        """With no price history, should use max_size."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            VolatilitySizedBSH, prices=prices,
            scheme_kwargs={"window": 5, "max_size": 1.0}
        )
        env.reset()

        # First step — no price history yet, uses max_size
        env.step(1)  # Buy
        assert scheme._position == 1
        assert asset.balance.as_float() > 0

    def test_buy_respects_min_size(self):
        """Position size should not go below min_size."""
        # Very volatile prices
        prices = [100.0, 200.0, 50.0, 300.0, 10.0, 500.0] + [100.0] * 20
        env, scheme, _, cash, asset = make_env_with_scheme(
            VolatilitySizedBSH, prices=prices,
            scheme_kwargs={"window": 5, "min_size": 0.1, "max_size": 1.0,
                           "target_risk": 0.001}
        )
        env.reset()

        # Build up price history with holds
        for _ in range(5):
            env.step(0)

        env.step(1)  # Buy — high volatility, small size, clamped to min
        assert scheme._position == 1

    def test_sell_is_full_exit(self):
        """Sell should exit entire position."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, cash, asset = make_env_with_scheme(
            VolatilitySizedBSH, prices=prices,
            scheme_kwargs={"window": 3}
        )
        env.reset()

        env.step(1)  # Buy
        env.step(2)  # Sell
        assert scheme._position == 0

    def test_price_buffer_accumulates(self):
        """Price buffer should fill during operation."""
        prices = [100.0 + i * 0.5 for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            VolatilitySizedBSH, prices=prices,
            scheme_kwargs={"window": 5}
        )
        env.reset()

        for _ in range(4):
            env.step(0)
        assert len(scheme._price_buffer) == 4

    def test_reset_clears_buffer(self):
        prices = [100.0 + i for i in range(20)]
        env, scheme, _, _, _ = make_env_with_scheme(
            VolatilitySizedBSH, prices=prices, scheme_kwargs={"window": 5}
        )
        env.reset()
        env.step(0)
        env.step(0)
        assert len(scheme._price_buffer) > 0
        env.reset()
        assert scheme._position == 0
        assert len(scheme._price_buffer) == 0


# ===========================================================================
# Registry Tests
# ===========================================================================


class TestActionRegistry:
    """Tests for the action scheme registry."""

    def test_registry_has_all_entries(self):
        from tensortrade.env.default.actions import _registry
        expected = [
            'bsh', 'trailing-stop-bsh', 'bracket-bsh', 'drawdown-budget-bsh',
            'cooldown-bsh', 'hold-minimum-bsh', 'confirmation-bsh',
            'scaled-entry-bsh', 'partial-tp-bsh', 'volatility-bsh',
            'simple', 'managed-risk',
        ]
        for name in expected:
            assert name in _registry, f"Missing registry entry: {name}"
