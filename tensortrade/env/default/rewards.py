from abc import abstractmethod

import numpy as np
import pandas as pd

from tensortrade.env.generic import RewardScheme, TradingEnv
from tensortrade.feed.core import DataFeed, Stream


class TensorTradeRewardScheme(RewardScheme):
    """An abstract base class for reward schemes for the default environment."""

    def reward(self, env: "TradingEnv") -> float:
        return self.get_reward(env.action_scheme.portfolio)

    @abstractmethod
    def get_reward(self, portfolio) -> float:
        """Gets the reward associated with current step of the episode.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio associated with the `TensorTradeActionScheme`.

        Returns
        -------
        float
            The reward for the current step of the episode.
        """
        raise NotImplementedError()


class SimpleProfit(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases
    in net worth.

    Parameters
    ----------
    window_size : int
        The size of the look back window for computing the reward.

    Attributes
    ----------
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(self, window_size: int = 1):
        self._window_size = self.default("window_size", window_size)
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

    def on_action(self, action: int) -> None:
        """Track action for stats."""
        if action == 1:
            self.buy_count += 1
        elif action == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        float
            The cumulative percentage change in net worth over the previous
            `window_size` time steps.
        """
        net_worths = [nw["net_worth"] for nw in portfolio.performance.values()]
        if len(net_worths) > 1:
            return net_worths[-1] / net_worths[-min(len(net_worths), self._window_size + 1)] - 1.0
        else:
            return 0.0

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        return {
            "trade_count": self.buy_count + self.sell_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
        }

    def reset(self) -> None:
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0


class RiskAdjustedReturns(TensorTradeRewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
    while penalizing more volatile strategies.

    Parameters
    ----------
    return_algorithm : {'sharpe', 'sortino'}, Default 'sharpe'.
        The risk-adjusted return metric to use.
    risk_free_rate : float, Default 0.
        The risk free rate of returns to use for calculating metrics.
    target_returns : float, Default 0
        The target returns per period for use in calculating the sortino ratio.
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(
        self,
        return_algorithm: str = "sharpe",
        risk_free_rate: float = 0.0,
        target_returns: float = 0.0,
        window_size: int = 1,
    ) -> None:
        algorithm = self.default("return_algorithm", return_algorithm)

        assert algorithm in ["sharpe", "sortino"]

        if algorithm == "sharpe":
            return_algorithm = self._sharpe_ratio
        elif algorithm == "sortino":
            return_algorithm = self._sortino_ratio

        self._return_algorithm = return_algorithm
        self._risk_free_rate = self.default("risk_free_rate", risk_free_rate)
        self._target_returns = self.default("target_returns", target_returns)
        self._window_size = self.default("window_size", window_size)
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

    def on_action(self, action: int) -> None:
        """Track action for stats."""
        if action == 1:
            self.buy_count += 1
        elif action == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        return {
            "trade_count": self.buy_count + self.sell_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
        }

    def reset(self) -> None:
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

    def _sharpe_ratio(self, returns: "pd.Series") -> float:
        """Computes the sharpe ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sharpe ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (np.std(returns) + 1e-9)

    def _sortino_ratio(self, returns: "pd.Series") -> float:
        """Computes the sortino ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sortino ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        # Only square the returns that are below target (downside)
        mask = returns < self._target_returns
        downside_returns[mask] = returns[mask] ** 2
        # Set non-downside returns to 0 for proper downside deviation calculation
        downside_returns[~mask] = 0

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.mean(downside_returns))

        return (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)

    def get_reward(self, portfolio: "Portfolio") -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.

        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.

        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worths = [nw["net_worth"] for nw in portfolio.performance.values()][-(self._window_size + 1) :]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)
        return risk_adjusted_return


class PBR(TensorTradeRewardScheme):
    """A reward scheme for position-based returns with commission awareness.

    * Let :math:`p_t` denote the price at time t.
    * Let :math:`x_t` denote the position at time t (0=cash, 1=long).
    * Let :math:`R_t` denote the reward at time t.

    The reward combines two components:

    1. **Directional signal**: :math:`(p_t - p_{t-1}) \\cdot x_t`
       - When long (x=1): rewarded for price going up
       - When cash (x=0): zero reward (neutral, not penalized)

    2. **Trade penalty**: :math:`-p_t \\cdot \\text{commission}` on trade actions
       - Scaled by current price so the penalty is in the same units as the
         price-based reward signal
       - A buy at price 100 with 1% commission costs 100 * 0.01 = 1.0 in reward,
         equivalent to 10 steps of +0.1 price gain — matching reality

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    commission : float
        The exchange commission rate. Used to penalize trades. Default 0.003.
    """

    registered_name = "pbr"

    def __init__(
        self,
        price: "Stream",
        commission: float = 0.003,
        trade_penalty_multiplier: float = 1.0,
        churn_penalty_multiplier: float = 0.75,
        churn_window: int = 4,
        hold_bonus: float = 0.0,
        reward_clip: float | None = None,
    ) -> None:
        super().__init__()
        self.position = 0  # Start in cash (0=flat, 1=long)
        self.commission = commission
        self.trade_penalty_multiplier = trade_penalty_multiplier
        self.churn_penalty_multiplier = churn_penalty_multiplier
        self.churn_window = max(0, int(churn_window))
        self.hold_bonus = hold_bonus
        self.reward_clip = reward_clip
        self._traded = False  # Whether current step involved a trade
        self._step_count = 0
        self._last_trade_step: int | None = None
        self._churn_trade_count = 0
        self._total_commission_penalty = 0.0
        self._total_reward = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")
        current_price = Stream.sensor(price, lambda p: p.value, dtype="float")

        reward = (position * r).fillna(0).rename("reward")

        self.feed = DataFeed([reward, current_price.rename("current_price")])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        self._traded = False
        if action == 1:
            if self.position == 0:
                self._traded = True
                self.buy_count += 1
            self.position = 1  # Buy → long
        elif action == 2:
            if self.position == 1:
                self._traded = True
                self.sell_count += 1
            self.position = 0  # Sell → cash (flat)
        else:
            self.hold_count += 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        data = self.feed.next()
        reward = data["reward"]
        self._step_count += 1

        # Penalize trades proportional to price × commission
        # This keeps the penalty in the same scale as price-based rewards
        if self._traded:
            current_price = data.get("current_price", 0)
            if current_price > 0:
                penalty = current_price * self.commission * self.trade_penalty_multiplier
                if (
                    self._last_trade_step is not None
                    and (self._step_count - self._last_trade_step) <= self.churn_window
                ):
                    penalty += current_price * self.commission * self.churn_penalty_multiplier
                    self._churn_trade_count += 1
                reward -= penalty
                self._total_commission_penalty += penalty
            self._last_trade_step = self._step_count
        elif self.hold_bonus:
            reward += self.hold_bonus

        if self.reward_clip is not None and self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self._total_reward += reward
        return reward

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        trade_count = self.buy_count + self.sell_count
        churn_ratio = (self._churn_trade_count / trade_count) if trade_count > 0 else 0.0
        avg_penalty = (self._total_commission_penalty / trade_count) if trade_count > 0 else 0.0
        return {
            "trade_count": trade_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
            "churn_trade_count": self._churn_trade_count,
            "churn_ratio": churn_ratio,
            "avg_penalty_per_trade": avg_penalty,
            "total_commission_penalty": self._total_commission_penalty,
            "cumulative_reward": self._total_reward,
        }

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = 0
        self._traded = False
        self._step_count = 0
        self._last_trade_step = None
        self._churn_trade_count = 0
        self._total_commission_penalty = 0.0
        self._total_reward = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        self.feed.reset()


class AdvancedPBR(TensorTradeRewardScheme):
    """An advanced reward scheme combining PBR with trading penalties and hold bonuses.

    This scheme aims to generate actual profits by:
    1. Rewarding position-based returns (like PBR)
    2. Penalizing excessive trading to reduce commission costs
    3. Rewarding holding during uncertain/sideways markets

    The reward formula is:
    R_t = pbr_weight * PBR + trade_penalty * |action_change| + hold_bonus * is_holding

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    pbr_weight : float
        Weight for the PBR component. Default 1.0.
    trade_penalty : float
        Penalty for changing position (should be negative). Default -0.001.
    hold_bonus : float
        Bonus for holding when conditions are uncertain. Default 0.0001.
    volatility_threshold : float
        Price change threshold below which market is considered "flat". Default 0.001.
    commission : float
        The exchange commission rate for commission-aware penalty. Default 0.003.
    """

    registered_name = "advanced-pbr"

    def __init__(
        self,
        price: "Stream",
        pbr_weight: float = 1.0,
        trade_penalty: float = -0.001,
        hold_bonus: float = 0.0001,
        volatility_threshold: float = 0.001,
        commission: float = 0.003,
        trade_penalty_multiplier: float = 1.0,
        churn_penalty_multiplier: float = 0.75,
        churn_window: int = 4,
        reward_clip: float | None = None,
    ) -> None:
        super().__init__()
        self.position = 0  # Start in cash (0=flat, 1=long)
        self.prev_action = 0
        self.prev_price = None
        self.pbr_weight = pbr_weight
        self.trade_penalty = trade_penalty
        self.hold_bonus = hold_bonus
        self.volatility_threshold = volatility_threshold
        self.commission = commission
        self.trade_penalty_multiplier = trade_penalty_multiplier
        self.churn_penalty_multiplier = churn_penalty_multiplier
        self.churn_window = max(0, int(churn_window))
        self.reward_clip = reward_clip
        self.action_changed = False
        self._traded = False
        self._step_count = 0
        self._last_trade_step: int | None = None
        self._churn_trade_count = 0
        self._total_commission_penalty = 0.0
        self._total_reward = 0.0

        # PBR component
        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")
        pbr_reward = (position * r).fillna(0).rename("pbr_reward")

        # Price stream for volatility calculation
        self.price_stream = Stream.sensor(price, lambda p: p.value, dtype="float")

        self.feed = DataFeed([pbr_reward, self.price_stream])
        self.feed.compile()

        self.trade_count = 0
        self.hold_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def on_action(self, action: int) -> None:
        self._traded = False
        # Track if action changed
        self.action_changed = action != self.prev_action

        self.prev_action = action
        if action == 1:
            if self.position == 0:
                self._traded = True
                self.buy_count += 1
            self.position = 1  # Buy → long
        elif action == 2:
            if self.position == 1:
                self._traded = True
                self.sell_count += 1
            self.position = 0  # Sell → cash (flat)
        else:
            # Count explicit hold actions, regardless of hold-bonus eligibility.
            self.hold_count += 1

        if self._traded:
            self.trade_count += 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        data = self.feed.next()
        pbr_reward = data["pbr_reward"]
        current_price = data.get(self.price_stream.name, 0)
        self._step_count += 1

        # 1. PBR component (scaled)
        reward = self.pbr_weight * pbr_reward

        # 2. Commission-aware trade penalty (price-scaled)
        if self._traded:
            if current_price > 0:
                penalty = current_price * self.commission * self.trade_penalty_multiplier
                if (
                    self._last_trade_step is not None
                    and (self._step_count - self._last_trade_step) <= self.churn_window
                ):
                    penalty += current_price * self.commission * self.churn_penalty_multiplier
                    self._churn_trade_count += 1
                reward -= penalty
                self._total_commission_penalty += penalty
            # Apply fixed trade penalty only when an order actually executes.
            reward += self.trade_penalty
            self._last_trade_step = self._step_count

        # 3. Hold bonus - reward for holding in flat/uncertain markets
        if self.prev_price is not None and current_price > 0:
            price_change = abs(current_price - self.prev_price) / self.prev_price
            is_flat_market = price_change < self.volatility_threshold

            # Only give hold bonus if we're holding (not trading) in a flat market
            if is_flat_market and not self._traded and self.prev_action == 0:
                reward += self.hold_bonus

        if self.reward_clip is not None and self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.prev_price = current_price
        self._total_reward += reward

        return reward

    def reset(self) -> None:
        """Resets the reward scheme state."""
        self.position = 0
        self.prev_action = 0
        self.prev_price = None
        self.action_changed = False
        self._traded = False
        self._step_count = 0
        self._last_trade_step = None
        self._churn_trade_count = 0
        self._total_commission_penalty = 0.0
        self._total_reward = 0.0
        self.trade_count = 0
        self.hold_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.feed.reset()

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        trade_count = self.buy_count + self.sell_count
        churn_ratio = (self._churn_trade_count / trade_count) if trade_count > 0 else 0.0
        avg_penalty = (self._total_commission_penalty / trade_count) if trade_count > 0 else 0.0
        return {
            "trade_count": trade_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
            "churn_trade_count": self._churn_trade_count,
            "churn_ratio": churn_ratio,
            "avg_penalty_per_trade": avg_penalty,
            "total_commission_penalty": self._total_commission_penalty,
            "cumulative_reward": self._total_reward,
        }


class FractionalPBR(TensorTradeRewardScheme):
    """Position-based returns for fractional positions (0.0 to 1.0).

    Unlike PBR which tracks a binary position (0=cash, 1=long), this scheme
    computes the actual position fraction from the portfolio each step.
    This makes it compatible with ALL action schemes, including Discrete(4)
    schemes like ScaledEntryBSH and PartialTakeProfitBSH.

    reward = price_diff * prev_position_fraction
             - |position_change| * price * commission

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    commission : float
        The exchange commission rate. Default 0.003.
    """

    registered_name = "fractional-pbr"

    def __init__(self, price: "Stream", commission: float = 0.003) -> None:
        super().__init__()
        self.commission = commission
        self._prev_position_frac = 0.0

        current_price = Stream.sensor(price, lambda p: p.value, dtype="float")
        price_diff = current_price.diff().fillna(0).rename("price_diff")

        self.feed = DataFeed([price_diff, current_price.rename("current_price")])
        self.feed.compile()

        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

    def on_action(self, action: int) -> None:
        """Track action for stats only. Position is read from portfolio."""
        if action == 1:
            self.buy_count += 1
        elif action == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        data = self.feed.next()
        price_diff = data["price_diff"]
        current_price = data["current_price"]

        # Compute position fraction from portfolio
        net_worth = portfolio.net_worth
        if net_worth and net_worth > 0:
            base_balance = portfolio.base_balance.as_float()
            position_frac = max(0.0, min(1.0, 1.0 - base_balance / net_worth))
        else:
            position_frac = 0.0

        # Reward: price movement weighted by previous position fraction
        reward = price_diff * self._prev_position_frac

        # Commission penalty proportional to position change
        position_change = abs(position_frac - self._prev_position_frac)
        if position_change > 0.001 and current_price > 0:
            reward -= current_price * self.commission * position_change

        self._prev_position_frac = position_frac
        return reward

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        return {
            "trade_count": self.buy_count + self.sell_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
        }

    def reset(self) -> None:
        self._prev_position_frac = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        self.feed.reset()


class MaxDrawdownPenalty(TensorTradeRewardScheme):
    """Reward that penalizes drawdown deepening.

    Each step the reward is::

        reward = (net_worth - prev_net_worth) / initial_net_worth
                 - penalty_weight * max(0, drawdown_t - drawdown_{t-1})

    where ``drawdown_t = (equity_peak - net_worth) / equity_peak``.

    This teaches the agent to avoid letting drawdown deepen, even when net
    worth is still rising.  Pairs naturally with DrawdownBudgetBSH.

    Compatible with ALL action schemes.

    Parameters
    ----------
    penalty_weight : float
        How strongly to penalize drawdown deepening. Default 2.0.
    """

    registered_name = "max-drawdown-penalty"

    def __init__(self, penalty_weight: float = 2.0) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight
        self._equity_peak = 0.0
        self._prev_net_worth = 0.0
        self._prev_drawdown = 0.0
        self._initial_net_worth = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

    def on_action(self, action: int) -> None:
        """Track action for stats."""
        if action == 1:
            self.buy_count += 1
        elif action == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        return {
            "trade_count": self.buy_count + self.sell_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
        }

    def get_reward(self, portfolio: "Portfolio") -> float:
        net_worth = portfolio.net_worth or 0.0

        # Initialize on first call
        if self._initial_net_worth == 0.0:
            self._initial_net_worth = net_worth
            self._prev_net_worth = net_worth
            self._equity_peak = net_worth
            return 0.0

        # Update equity peak
        self._equity_peak = max(self._equity_peak, net_worth)

        # Current drawdown fraction
        if self._equity_peak > 0:
            drawdown = (self._equity_peak - net_worth) / self._equity_peak
        else:
            drawdown = 0.0

        # Base reward: normalized net worth change
        if self._initial_net_worth > 0:
            reward = (net_worth - self._prev_net_worth) / self._initial_net_worth
        else:
            reward = 0.0

        # Drawdown penalty: only when drawdown is deepening
        drawdown_increase = max(0.0, drawdown - self._prev_drawdown)
        reward -= self.penalty_weight * drawdown_increase

        self._prev_net_worth = net_worth
        self._prev_drawdown = drawdown
        return reward

    def reset(self) -> None:
        self._equity_peak = 0.0
        self._prev_net_worth = 0.0
        self._prev_drawdown = 0.0
        self._initial_net_worth = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0


class AdaptiveProfitSeeker(TensorTradeRewardScheme):
    """Sophisticated reward that strongly incentivizes profitable trading.

    Compatible with ALL action schemes including ScaledEntryBSH.

    Design principle: growth signal DOMINATES all other components so the
    agent never learns that "do nothing" is optimal. Auxiliary signals
    provide shaping but can never outweigh actual profit.

    Components (in priority order):
    1. Net worth growth (dominant signal, weight=1.0, raw percentage)
    2. Market participation bonus: positive reward for having a position
    3. Momentum bonus: extra for being right about direction
    4. Mild drawdown penalty: only kicks in on large drawdown deepening

    Parameters
    ----------
    price : `Stream`
        The price stream.
    participation_bonus : float
        Per-step bonus for being in the market. Default 0.0002.
    momentum_weight : float
        Weight for directional momentum bonus. Default 2.0.
    drawdown_weight : float
        Penalty weight for drawdown deepening. Default 0.5.
    commission : float
        Commission rate for trade cost. Default 0.0001.
    """

    registered_name = "adaptive-profit-seeker"

    def __init__(
        self,
        price: "Stream",
        participation_bonus: float = 0.0002,
        momentum_weight: float = 2.0,
        drawdown_weight: float = 0.5,
        commission: float = 0.0001,
    ) -> None:
        super().__init__()
        self.participation_bonus = participation_bonus
        self.momentum_weight = momentum_weight
        self.drawdown_weight = drawdown_weight
        self.commission = commission

        current_price = Stream.sensor(price, lambda p: p.value, dtype="float")
        price_return = current_price.pct_change().fillna(0).rename("price_return")
        self.feed = DataFeed([price_return, current_price.rename("current_price")])
        self.feed.compile()

        self._prev_net_worth = 0.0
        self._equity_peak = 0.0
        self._prev_drawdown = 0.0
        self._prev_position_frac = 0.0

        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

    def on_action(self, action: int) -> None:
        """Track action for stats."""
        if action == 1:
            self.buy_count += 1
        elif action == 2:
            self.sell_count += 1
        else:
            self.hold_count += 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        self.feed.next()  # advance price feed

        net_worth = portfolio.net_worth or 0.0

        # Initialize on first call
        if self._prev_net_worth == 0.0:
            self._prev_net_worth = net_worth
            self._equity_peak = net_worth
            return 0.0

        # Position fraction from portfolio
        base_balance = portfolio.base_balance.as_float()
        position_frac = max(0.0, min(1.0, 1.0 - base_balance / net_worth)) if net_worth > 0 else 0.0

        # === CORE: SimpleProfit (this is what actually works) ===
        nw_return = (net_worth / self._prev_net_worth - 1.0) if self._prev_net_worth > 0 else 0.0

        # === ONLY ADDITION: tiny participation bonus ===
        # Breaks the tie between "do nothing" and "trade" when growth is ~0.
        # Must be tiny enough to never outweigh actual losses.
        participation = self.participation_bonus * position_frac

        reward = nw_return + participation

        # Update state
        self._prev_net_worth = net_worth
        self._prev_position_frac = position_frac

        return reward

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        return {
            "trade_count": self.buy_count + self.sell_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
        }

    def reset(self) -> None:
        self._prev_net_worth = 0.0
        self._equity_peak = 0.0
        self._prev_drawdown = 0.0
        self._prev_position_frac = 0.0
        self._step_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        self.feed.reset()


_registry = {
    "simple": SimpleProfit,
    "risk-adjusted": RiskAdjustedReturns,
    "pbr": PBR,
    "advanced-pbr": AdvancedPBR,
    "fractional-pbr": FractionalPBR,
    "max-drawdown-penalty": MaxDrawdownPenalty,
    "adaptive-profit-seeker": AdaptiveProfitSeeker,
}


def get(identifier: str) -> "TensorTradeRewardScheme":
    """Gets the `RewardScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `RewardScheme`

    Returns
    -------
    `TensorTradeRewardScheme`
        The reward scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry:
        msg = f"Identifier {identifier} is not associated with any `RewardScheme`."
        raise KeyError(msg)
    return _registry[identifier]()
