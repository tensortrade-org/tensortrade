
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Callable

from tensortrade.env.generic import RewardScheme, TradingEnv


class TensorTradeRewardScheme(RewardScheme):

    def reward(self, env: TradingEnv):
        return self.get_reward(env.action_scheme.portfolio)

    @abstractmethod
    def get_reward(self, portfolio):
        raise NotImplementedError()


class SimpleProfit(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size: int = 1):
        self.window_size = self.default('window_size', window_size)

    def reset(self):
        pass

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a sliding window.

        Args:
            portfolio: The portfolio being used by the environment.

        Returns:
            The cumulative percentage change in net worth over the previous `window_size` timesteps.
        """
        returns = portfolio.performance['net_worth'].pct_change().dropna()
        returns = (1 + returns[-self.window_size:]).cumprod() - 1
        return 0 if len(returns) < 1 else returns.iloc[-1]


class RiskAdjustedReturns(TensorTradeRewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
    while penalizing more volatile strategies.
    """

    def __init__(self,
                 return_algorithm: str = 'sharpe',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1):
        """
        Args:
            return_algorithm (optional): The risk-adjusted return metric to use. Options are 'sharpe' and 'sortino'. Defaults to 'sharpe'.
            risk_free_rate (optional): The risk free rate of returns to use for calculating metrics. Defaults to 0.
            target_returns (optional): The target returns per period for use in calculating the sortino ratio. Default to 0.
        """
        algorithm = self.default('return_algorithm', return_algorithm)

        self._return_algorithm = self._return_algorithm_from_str(algorithm)
        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)

    def _return_algorithm_from_str(self, algorithm_str: str) -> Callable[[pd.DataFrame], float]:
        assert algorithm_str in ['sharpe', 'sortino']

        if algorithm_str == 'sharpe':
            return self._sharpe_ratio
        elif algorithm_str == 'sortino':
            return self._sortino_ratio

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Return the sharpe ratio for a given series of a returns.

        References:
            - https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1E-9) / (np.std(returns) + 1E-9)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Return the sortino ratio for a given series of a returns.

        References:
            - https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate + 1E-9) / (downside_std + 1E-9)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Return the reward corresponding to the selected risk-adjusted return metric."""
        returns = portfolio.performance['net_worth'][-(self._window_size + 1):].pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return


_registry = {
    'simple': SimpleProfit,
    'risk-adjusted': RiskAdjustedReturns
}


def get(identifier: str) -> TensorTradeRewardScheme:
    """Gets the `RewardScheme` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `RewardScheme`

    Raises:
        KeyError: if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry.keys():
        raise KeyError(f"Identifier {identifier} is not associated with any `RewardScheme`.")
    return _registry[identifier]()
