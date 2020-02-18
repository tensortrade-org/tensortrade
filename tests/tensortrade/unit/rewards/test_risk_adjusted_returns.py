import pytest
import numpy as np
import pandas as pd

from tensortrade.rewards import RiskAdjustedReturns


@pytest.fixture
def net_worths():
    return pd.Series([100, 400, 350, 450, 200, 400, 330, 560], name="net_worth")


class TestRiskAdjustedReturns:

    def test_sharpe_ratio(self, net_worths):
        scheme = RiskAdjustedReturns(return_algorithm='sharpe',
                                     risk_free_rate=0,
                                     window_size=1)

        returns = net_worths[-2:].pct_change().dropna()

        expected_ratio = (np.mean(returns) + 1E-9) / (np.std(returns) + 1E-9)
        sharpe_ratio = scheme._sharpe_ratio(returns)

        assert sharpe_ratio == expected_ratio

    def test_sortino_ratio(self, net_worths):
        scheme = RiskAdjustedReturns(return_algorithm='sortino',
                                     risk_free_rate=0,
                                     target_returns=0,
                                     window_size=1)

        returns = net_worths[-2:].pct_change().dropna()

        downside_returns = returns.copy()
        downside_returns[returns < 0] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        expected_ratio = (expected_return + 1E-9) / (downside_std + 1E-9)
        sortino_ratio = scheme._sortino_ratio(returns)

        assert sortino_ratio == expected_ratio
