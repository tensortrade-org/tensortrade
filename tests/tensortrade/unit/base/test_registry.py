
import warnings

import tensortrade.actions as actions
import tensortrade.rewards as rewards

from tensortrade.actions import SimpleOrders, ManagedRiskOrders

warnings.filterwarnings("ignore")


def test_dynamic_actions():
    assert isinstance(actions.get('simple'), SimpleOrders)


def test_managed_risk_actions():
    assert isinstance(actions.get('managed-risk'), ManagedRiskOrders)


def test_simple_reward_scheme():
    assert isinstance(rewards.get('simple'), rewards.SimpleProfit)
