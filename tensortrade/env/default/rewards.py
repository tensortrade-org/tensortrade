
from abc import abstractmethod

import numpy as np
import pandas as pd

from tensortrade.env.generic import RewardScheme, TradingEnv
from tensortrade.feed.core import Stream, DataFeed
import math


class TensorTradeRewardScheme(RewardScheme):
    """An abstract base class for reward schemes for the default environment.
    """

    def reward(self, env: 'TradingEnv') -> float:
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
        self._window_size = self.default('window_size', window_size)

    def get_reward(self, portfolio: 'Portfolio') -> float:
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
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()]
        returns = [(b - a) / a for a, b in zip(net_worths[::1], net_worths[1::1])]
        returns = np.array([x + 1 for x in returns[-self._window_size:]]).cumprod() - 1
        return 0 if len(returns) < 1 else returns[-1]


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

    def __init__(self,
                 return_algorithm: str = 'sharpe',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1) -> None:
        algorithm = self.default('return_algorithm', return_algorithm)

        assert algorithm in ['sharpe', 'sortino']

        if algorithm == 'sharpe':
            return_algorithm = self._sharpe_ratio
        elif algorithm == 'sortino':
            return_algorithm = self._sortino_ratio

        self._return_algorithm = return_algorithm
        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)

    def _sharpe_ratio(self, returns: 'pd.Series') -> float:
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

    def _sortino_ratio(self, returns: 'pd.Series') -> float:
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
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)

    def get_reward(self, portfolio: 'Portfolio') -> float:
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
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()][-(self._window_size + 1):]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)
        return risk_adjusted_return


class PBR(TensorTradeRewardScheme):
    """A reward scheme for position-based returns.

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

    def __init__(self, price: 'Stream') -> None:
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (position * r).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio') -> float:
        return self.feed.next()["reward"]

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = -1
        self.feed.reset()


_registry = {
    'simple': SimpleProfit,
    'risk-adjusted': RiskAdjustedReturns
}


def get(identifier: str) -> 'TensorTradeRewardScheme':
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
