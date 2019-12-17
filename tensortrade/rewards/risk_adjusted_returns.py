# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import pandas as pd
import numpy as np

from typing import Callable

from tensortrade.rewards import RewardScheme


class RiskAdjustedReturns(RewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth, while penalizing more volatile strategies.
    """

    def __init__(self, return_algorithm: str = 'sharpe', risk_free_rate: float = 0., target_returns: float = 0.):
        """
        Args:
            return_algorithm (optional): The risk-adjusted return metric to use. Options are 'sharpe' and 'sortino'. Defaults to 'sharpe'.
            risk_free_rate (optional): The risk free rate of returns to use for calculating metrics. Defaults to 0.
            target_returns (optional): The target returns per period for use in calculating the sortino ratio. Default to 0.
        """
        self._return_algorithm = self._return_algorithm_from_str(return_algorithm)
        self._risk_free_rate = risk_free_rate
        self._target_returns = target_returns

    def _return_algorithm_from_str(self, algorithm_str: str) -> Callable[[pd.DataFrame], float]:
        if algorithm_str == 'sharpe':
            return self._sharpe_ratio
        elif algorithm_str == 'sortino':
            return self._sortino_ratio

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Return the sharpe ratio for a given series of a returns.

        https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (returns.mean() - self._risk_free_rate) / (returns.std() + 1E-9)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Return the sortino ratio for a given series of a returns.

        https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()

        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = returns.mean()
        downside_std = np.sqrt(downside_returns.std())

        return (expected_return - self._risk_free_rate) / (downside_std + 1E-9)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Return the reward corresponding to the selected risk-adjusted return metric."""
        returns = portfolio.performance['net_worth'].diff()

        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return
