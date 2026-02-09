
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import tensortrade.env.default.rewards as rewards
from tensortrade.core import TradingContext
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import BTC, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet


class TestTensorTradeRewardScheme:

    def setup_method(self):
        self.config = {
            'base_instrument': 'USD',
            'instruments': 'ETH',
            'rewards': {
                'size': 0
            }
        }

        class ConcreteRewardScheme(rewards.TensorTradeRewardScheme):

            def get_reward(self, env) -> float:
                pass

        self.concrete_class = ConcreteRewardScheme

    def test_injects_reward_scheme_with_context(self):

        with TradingContext(self.config):

            reward_scheme = self.concrete_class()

            assert hasattr(reward_scheme.context, 'size')
            assert reward_scheme.context.size == 0
            assert reward_scheme.context['size'] == 0

    def test_injects_string_initialized_reward_scheme(self):

        with TradingContext(self.config):

            reward_scheme = rewards.get('simple')

            assert reward_scheme.registered_name == "rewards"
            assert hasattr(reward_scheme.context, 'size')
            assert reward_scheme.context.size == 0
            assert reward_scheme.context['size'] == 0


@pytest.fixture
def net_worths():
    return pd.Series([100, 400, 350, 450, 200, 400, 330, 560], name="net_worth")

def net_worths_to_dict(net_worths):
    result = OrderedDict()
    index = 0
    for i in net_worths:
        result[index] = {"net_worth": i}
        index += 1
    return result


def stream_net_worths(net_worths):
    index = 0
    for i in net_worths:
        yield index, {"net_worth": i}
        index += 1


class TestSimpleProfit:

    def test_get_reward(self, net_worths):
        portfolio = Portfolio(USD)
        portfolio._performance = net_worths_to_dict(net_worths)

        pct_chg = net_worths.pct_change()

        reward_scheme = rewards.SimpleProfit()
        assert reward_scheme.get_reward(portfolio) == pct_chg.iloc[-1]  # default window size 1

        reward_scheme._window_size = 3
        reward = ((1 + pct_chg.iloc[-1]) * (1 + pct_chg.iloc[-2]) * (1 + pct_chg.iloc[-3])) - 1
        assert reward_scheme.get_reward(portfolio) == reward


class TestRiskAdjustedReturns:

    def test_sharpe_ratio(self, net_worths):
        scheme = rewards.RiskAdjustedReturns(
            return_algorithm='sharpe',
            risk_free_rate=0,
            window_size=1
        )

        returns = net_worths[-2:].pct_change().dropna()

        expected_ratio = (np.mean(returns) + 1E-9) / (np.std(returns) + 1E-9)
        sharpe_ratio = scheme._sharpe_ratio(returns)

        assert sharpe_ratio == expected_ratio

    def test_sortino_ratio(self, net_worths):
        scheme = rewards.RiskAdjustedReturns(
            return_algorithm='sortino',
            risk_free_rate=0,
            target_returns=0,
            window_size=1
        )

        returns = net_worths[-2:].pct_change().dropna()

        downside_returns = returns.copy()
        downside_returns[returns < 0] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        expected_ratio = (expected_return + 1E-9) / (downside_std + 1E-9)
        sortino_ratio = scheme._sortino_ratio(returns)

        assert sortino_ratio == expected_ratio


class TestAdvancedPBR:
    """Tests for AdvancedPBR reward/metrics accounting semantics."""

    @staticmethod
    def _make_env(prices=None, **kwargs):
        import tensortrade.env.default as default
        from tensortrade.env.default.actions import BSH

        if prices is None:
            prices = [100.0, 101.0, 102.0, 103.0]
        price_stream = Stream.source(prices, dtype="float").rename("USD-BTC")

        exchange = Exchange(
            "exchange",
            service=execute_order,
            options=ExchangeOptions(commission=0.0),
        )(price_stream)

        cash = Wallet(exchange, 10000.0 * USD)
        asset = Wallet(exchange, 0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        features = [Stream.source(prices, dtype="float").rename("close")]
        feed = DataFeed(features)
        feed.compile()

        reward_scheme = rewards.AdvancedPBR(price=price_stream, **kwargs)
        action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)
        env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            window_size=1,
            max_allowed_loss=0.99,
        )
        env.reset()
        return env, reward_scheme

    def test_trade_count_and_penalty_only_on_executed_trade(self):
        env, scheme = self._make_env(
            pbr_weight=0.0,
            commission=0.0,
            trade_penalty=-0.5,
            hold_bonus=0.0,
        )

        env.step(1)  # Executed buy (flat -> long)
        env.step(0)  # Hold
        _, reward, _, _, _ = env.step(1)  # No-op buy (already long), action_changed=True

        stats = scheme.get_stats()
        assert stats["trade_count"] == 1
        assert stats["buy_count"] == 1
        assert stats["sell_count"] == 0
        assert stats["hold_count"] == 1
        # No executed trade on this step => no fixed trade penalty applied.
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_hold_count_tracks_hold_actions_even_without_bonus(self):
        env, scheme = self._make_env(
            prices=[100.0, 101.0, 102.0, 103.0],
            pbr_weight=0.0,
            commission=0.0,
            trade_penalty=0.0,
            hold_bonus=1.0,
            volatility_threshold=0.0,  # Never treated as flat when price moves.
        )

        env.step(0)
        _, reward, _, _, _ = env.step(0)

        stats = scheme.get_stats()
        assert stats["hold_count"] == 2
        # Hold bonus is not eligible, but hold actions are still counted.
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_hold_bonus_not_applied_to_non_hold_noop_actions(self):
        env, scheme = self._make_env(
            prices=[100.0, 100.0, 100.0, 100.0, 100.0],
            pbr_weight=0.0,
            commission=0.0,
            trade_penalty=0.0,
            hold_bonus=0.25,
            volatility_threshold=1.0,  # Flat market.
        )

        env.step(1)  # Executed buy
        _, hold_reward, _, _, _ = env.step(0)  # Hold in flat market => gets hold bonus
        assert hold_reward == pytest.approx(0.25, abs=1e-9)

        _, reward, _, _, _ = env.step(1)  # No-op buy in flat market (not a hold action)
        assert reward == pytest.approx(0.0, abs=1e-9)


class TestFractionalPBR:
    """Tests for FractionalPBR reward scheme."""

    def _make_env_components(self, n_steps=10, initial_cash=10000.0):
        """Create minimal env components for testing FractionalPBR."""
        prices = [100.0 + i * 0.5 for i in range(n_steps)]
        price = Stream.source(prices, dtype="float").rename("USD-BTC")

        exchange = Exchange(
            "exchange", service=execute_order,
            options=ExchangeOptions(commission=0.001)
        )(price)

        cash = Wallet(exchange, initial_cash * USD)
        asset = Wallet(exchange, 0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        return price, portfolio, cash, asset

    def test_init_defaults(self):
        """FractionalPBR should initialize with zero position fraction."""
        price, _, _, _ = self._make_env_components()
        fpbr = rewards.FractionalPBR(price=price)
        assert fpbr._prev_position_frac == 0.0
        assert fpbr.commission == 0.003

    def test_custom_commission(self):
        """FractionalPBR should accept custom commission."""
        price, _, _, _ = self._make_env_components()
        fpbr = rewards.FractionalPBR(price=price, commission=0.01)
        assert fpbr.commission == 0.01

    def test_on_action_stats(self):
        """on_action should track buy/sell/hold counts for stats."""
        price, _, _, _ = self._make_env_components()
        fpbr = rewards.FractionalPBR(price=price)

        fpbr.on_action(0)  # hold
        fpbr.on_action(1)  # buy
        fpbr.on_action(1)  # buy
        fpbr.on_action(2)  # sell

        stats = fpbr.get_stats()
        assert stats["buy_count"] == 2
        assert stats["sell_count"] == 1
        assert stats["hold_count"] == 1
        assert stats["trade_count"] == 3

    def test_reset(self):
        """Reset should clear all state."""
        price, _, _, _ = self._make_env_components()
        fpbr = rewards.FractionalPBR(price=price)

        fpbr.on_action(1)
        fpbr._prev_position_frac = 0.5

        fpbr.reset()
        assert fpbr._prev_position_frac == 0.0
        assert fpbr.buy_count == 0
        assert fpbr.sell_count == 0
        assert fpbr.hold_count == 0

    def test_reward_via_env(self):
        """FractionalPBR should return zero reward when holding only cash."""
        import tensortrade.env.default as default
        from tensortrade.env.default.actions import BSH

        n_steps = 10
        prices = [100.0 + i * 0.5 for i in range(n_steps)]
        price = Stream.source(prices, dtype="float").rename("USD-BTC")

        exchange = Exchange(
            "exchange", service=execute_order,
            options=ExchangeOptions(commission=0.0)
        )(price)

        cash = Wallet(exchange, 10000.0 * USD)
        asset = Wallet(exchange, 0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        features = [Stream.source(prices, dtype="float").rename("close")]
        feed = DataFeed(features)
        feed.compile()

        fpbr = rewards.FractionalPBR(price=price, commission=0.0)
        action_scheme = BSH(cash=cash, asset=asset).attach(fpbr)

        env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=fpbr,
            window_size=1,
            max_allowed_loss=0.99,
        )

        # Step with hold action — should get ~0 reward (all cash, no exposure)
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_registry(self):
        """FractionalPBR should be in the registry."""
        assert 'fractional-pbr' in rewards._registry
        assert rewards._registry['fractional-pbr'] == rewards.FractionalPBR


class TestMaxDrawdownPenalty:
    """Tests for MaxDrawdownPenalty reward scheme."""

    def _make_portfolio_with_net_worth(self, net_worth):
        """Create a portfolio mock with a specific net worth."""
        portfolio = Portfolio(USD)
        portfolio._net_worth = net_worth
        portfolio._initial_net_worth = net_worth
        return portfolio

    def test_init_defaults(self):
        """MaxDrawdownPenalty should initialize with default penalty weight."""
        scheme = rewards.MaxDrawdownPenalty()
        assert scheme.penalty_weight == 2.0

    def test_custom_penalty_weight(self):
        """MaxDrawdownPenalty should accept custom penalty weight."""
        scheme = rewards.MaxDrawdownPenalty(penalty_weight=5.0)
        assert scheme.penalty_weight == 5.0

    def test_first_call_returns_zero(self):
        """First call should return 0 (initialization step)."""
        scheme = rewards.MaxDrawdownPenalty()
        portfolio = self._make_portfolio_with_net_worth(10000.0)
        assert scheme.get_reward(portfolio) == 0.0

    def test_positive_return_no_drawdown(self):
        """Rising net worth with no drawdown should give positive reward."""
        scheme = rewards.MaxDrawdownPenalty()
        p = self._make_portfolio_with_net_worth(10000.0)

        scheme.get_reward(p)  # init

        p._net_worth = 10100.0
        reward = scheme.get_reward(p)
        # (10100 - 10000) / 10000 = 0.01, no drawdown penalty
        assert reward == pytest.approx(0.01, abs=1e-6)

    def test_drawdown_deepening_penalized(self):
        """Deepening drawdown should reduce the reward."""
        scheme = rewards.MaxDrawdownPenalty(penalty_weight=2.0)
        p = self._make_portfolio_with_net_worth(10000.0)

        scheme.get_reward(p)  # init

        # Price goes up first (establish peak)
        p._net_worth = 10500.0
        scheme.get_reward(p)

        # Now price drops — drawdown deepens
        p._net_worth = 10000.0
        reward = scheme.get_reward(p)

        # base reward = (10000 - 10500) / 10000 = -0.05
        # drawdown = (10500 - 10000) / 10500 ≈ 0.04762
        # drawdown increase from 0 → 0.04762
        # penalty = 2.0 * 0.04762 ≈ 0.09524
        # total = -0.05 - 0.09524 ≈ -0.14524
        assert reward < -0.05  # strictly worse than just the net worth drop

    def test_recovery_not_penalized(self):
        """Recovery (drawdown shrinking) should not incur penalty."""
        scheme = rewards.MaxDrawdownPenalty(penalty_weight=2.0)
        p = self._make_portfolio_with_net_worth(10000.0)

        scheme.get_reward(p)  # init

        # Drop
        p._net_worth = 9000.0
        scheme.get_reward(p)

        # Partial recovery — drawdown shrinks
        p._net_worth = 9500.0
        reward = scheme.get_reward(p)

        # base reward = (9500 - 9000) / 10000 = 0.05
        # drawdown shrinks from ~0.10 to ~0.05 → no penalty (drawdown_increase = 0)
        assert reward == pytest.approx(0.05, abs=1e-6)

    def test_reset(self):
        """Reset should clear all state."""
        scheme = rewards.MaxDrawdownPenalty()
        p = self._make_portfolio_with_net_worth(10000.0)

        scheme.get_reward(p)
        p._net_worth = 10500.0
        scheme.get_reward(p)

        scheme.reset()
        assert scheme._equity_peak == 0.0
        assert scheme._prev_net_worth == 0.0
        assert scheme._prev_drawdown == 0.0
        assert scheme._initial_net_worth == 0.0

    def test_registry(self):
        """MaxDrawdownPenalty should be in the registry."""
        assert 'max-drawdown-penalty' in rewards._registry
        assert rewards._registry['max-drawdown-penalty'] == rewards.MaxDrawdownPenalty
