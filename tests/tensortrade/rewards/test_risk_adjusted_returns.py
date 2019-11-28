import pytest
import numpy as np
import pandas as pd

from tensortrade.rewards import RiskAdjustedReturns


@pytest.fixture
def net_worths():
    return pd.DataFrame([{
        'net_worth': 100
    }, {
        'net_worth': 400
    },
        {
        'net_worth': 350
    }])


class TestRiskAdjustedReturns:

    def test_sharpe_ratio(self, net_worths):
        scheme = RiskAdjustedReturns(return_algorithm='sharpe', risk_free_rate=0)

        returns = net_worths['net_worth'].diff()

        sharpe_ratio = scheme._sharpe_ratio(returns)

        expected_ratio = returns.mean() / (returns.std() + 1E-9)

        assert sharpe_ratio == expected_ratio

    def test_sortino_ratio(self, net_worths):
        scheme = RiskAdjustedReturns(
            return_algorithm='sortino', risk_free_rate=0, target_returns=0)

        returns = net_worths['net_worth'].diff()

        sortino_ratio = scheme._sortino_ratio(returns)

        downside_returns = pd.Series([0])

        returns[returns < 0] = returns ** 2

        expected_return = returns.mean()
        downside_std = np.sqrt(downside_returns.mean())

        expected_ratio = expected_return / (downside_std + 1E-9)

        assert sortino_ratio == expected_ratio
