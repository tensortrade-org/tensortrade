import pytest
import numpy as np
import pandas as pd

from tensortrade.rewards import RiskAdjustedReturns


@pytest.fixture
def net_worths():
    return pd.Series([100, 400, 350, 450, 200, 400, 330, 560], name="net_worth")


class TestRiskAdjustedReturns:

    def test_sharpe_ratio(self, net_worths):
        scheme = RiskAdjustedReturns(return_algorithm='sharpe', risk_free_rate=0)

        returns = net_worths.diff()

        sharpe_ratio = scheme._sharpe_ratio(returns)

        expected_ratio = returns.mean() / (returns.std() + 1E-9)

        assert sharpe_ratio == expected_ratio

    @pytest.mark.skip("Needs to be reevaluated.")
    def test_sortino_ratio(self, net_worths):
        scheme = RiskAdjustedReturns(
            return_algorithm='sortino',
            risk_free_rate=0,
            target_returns=0
        )

        returns = net_worths.diff()

        sortino_ratio = scheme._sortino_ratio(returns)

        expected_return = returns.mean()
        downside_std = returns[returns < 0].std()

        expected_ratio = expected_return / downside_std

        assert sortino_ratio == expected_ratio
