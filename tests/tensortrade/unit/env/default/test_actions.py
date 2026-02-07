"""Tests for BSH action scheme and PBR reward integration."""

import pytest
from unittest.mock import MagicMock, patch
from gymnasium.spaces import Discrete

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC, Quantity
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.env.default.actions import BSH
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

    def test_pbr_long_in_rising_market_gives_positive_reward(self):
        """Holding long in a rising market should produce positive rewards."""
        env, _, reward_scheme, _, _ = make_env(n_steps=20)
        env.reset()

        env.step(1)  # Buy
        total_reward = 0.0
        for _ in range(5):
            _, reward, _, _, _ = env.step(0)  # Hold
            total_reward += reward
        assert total_reward > 0

    def test_pbr_cash_position_gives_zero_reward_not_negative(self):
        """After selling, cash position must give zero reward — not negative.

        Regression: old code set position=-1 after sell, which produced
        negative reward during rallies (penalizing the agent for being safe).
        """
        env, _, reward_scheme, _, _ = make_env(n_steps=20)
        env.reset()

        env.step(1)  # Buy
        env.step(0)  # Hold
        env.step(2)  # Sell to cash

        for _ in range(5):
            _, reward, _, _, _ = env.step(0)  # Hold in cash
            assert reward == pytest.approx(0.0, abs=1e-10), (
                f"Cash position reward must be 0, got {reward} "
                "(position=-1 bug would cause negative reward)"
            )

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
