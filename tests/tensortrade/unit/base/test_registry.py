
import warnings

import tensortrade.actions as actions
import tensortrade.rewards as rewards

from tensortrade.actions import SimpleOrders, ManagedRiskOrders
from tensortrade.rewards import SimpleProfit, RiskAdjustedReturns

warnings.filterwarnings("ignore")


def test_simple_actions():
    assert isinstance(actions.get('simple'), SimpleOrders)


def test_managed_risk_actions():
    assert isinstance(actions.get('managed-risk'), ManagedRiskOrders)


def test_simple_reward_scheme():
    assert isinstance(rewards.get('simple'), SimpleProfit)


def test_risk_adjusted_reward_scheme():
    assert isinstance(rewards.get('risk-adjusted'), RiskAdjustedReturns)
