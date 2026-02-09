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
            return (
                net_worths[-1]
                / net_worths[-min(len(net_worths), self._window_size + 1)]
                - 1.0
            )
        else:
            return 0.0


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
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (
            np.std(returns) + 1e-9
        )

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
        net_worths = [nw["net_worth"] for nw in portfolio.performance.values()][
            -(self._window_size + 1) :
        ]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)
        return risk_adjusted_return


class PBR(TensorTradeRewardScheme):
    r"""A reward scheme for position-based returns.

    * Let :math:`p_t` denote the price at time t.
    * Let :math:`x_t` denote the position at time t.
    * Let :math:`R_t` denote the reward at time t.

    Then the reward is defined as,
    :math:`R_{t} = (p_{t} - p_{t-1}) \cdot x_{t}`.

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    """

    registered_name = "pbr"

    def __init__(self, price: "Stream") -> None:
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (position * r).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        return self.feed.next()["reward"]

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = -1
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
    """

    registered_name = "advanced-pbr"

    def __init__(
        self,
        price: "Stream",
        pbr_weight: float = 1.0,
        trade_penalty: float = -0.001,
        hold_bonus: float = 0.0001,
        volatility_threshold: float = 0.001,
    ) -> None:
        super().__init__()
        self.position = -1
        self.prev_action = 0
        self.prev_price = None
        self.pbr_weight = pbr_weight
        self.trade_penalty = trade_penalty
        self.hold_bonus = hold_bonus
        self.volatility_threshold = volatility_threshold

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

    def on_action(self, action: int) -> None:
        # Track if action changed
        self.action_changed = action != self.prev_action
        if self.action_changed:
            self.trade_count += 1

        self.prev_action = action
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: "Portfolio") -> float:
        data = self.feed.next()
        pbr_reward = data["pbr_reward"]
        current_price = data.get(self.price_stream.name, 0)

        # 1. PBR component (scaled)
        reward = self.pbr_weight * pbr_reward

        # 2. Trading penalty - penalize changing positions
        if self.action_changed:
            reward += self.trade_penalty

        # 3. Hold bonus - reward for holding in flat/uncertain markets
        if self.prev_price is not None and current_price > 0:
            price_change = abs(current_price - self.prev_price) / self.prev_price
            is_flat_market = price_change < self.volatility_threshold

            # Only give hold bonus if we're holding (not trading) in a flat market
            if is_flat_market and not self.action_changed:
                reward += self.hold_bonus
                self.hold_count += 1

        self.prev_price = current_price

        return reward

    def reset(self) -> None:
        """Resets the reward scheme state."""
        self.position = -1
        self.prev_action = 0
        self.prev_price = None
        self.action_changed = False
        self.trade_count = 0
        self.hold_count = 0
        self.feed.reset()

    def get_stats(self) -> dict:
        """Returns trading statistics for analysis."""
        return {"trade_count": self.trade_count, "hold_count": self.hold_count}


_registry = {
    "simple": SimpleProfit,
    "risk-adjusted": RiskAdjustedReturns,
    "pbr": PBR,
    "advanced-pbr": AdvancedPBR,
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
    if identifier not in _registry.keys():
        msg = f"Identifier {identifier} is not associated with any `RewardScheme`."
        raise KeyError(msg)
    return _registry[identifier]()
