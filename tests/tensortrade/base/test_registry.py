
import pytest
import warnings
import unittest.mock as mock

import tensortrade.actions as actions
import tensortrade.exchanges as exchanges
import tensortrade.rewards as rewards
import tensortrade.slippage as slippage
import tensortrade.environments as envs

from tensortrade.environments import TradingEnvironment
from tensortrade.actions import DynamicOrders, ManagedRiskOrders

warnings.filterwarnings("ignore")


def test_dynamic_actions():
    assert isinstance(actions.get('dynamic'), DynamicOrders)


def test_managed_risk_actions():
    assert isinstance(actions.get('managed-risk'), ManagedRiskOrders)


def test_simple_reward_scheme():
    assert isinstance(rewards.get('simple'), SimpleProfit)


def test_random_uniform_slippage_model():
    assert isinstance(slippage.get('uniform'), RandomUniformSlippageModel)


def make_env(exchange: str, action: str, reward: str):
    portfolio = mock.Mock()
    feed = mock.Mock()
    return TradingEnvironment(portfolio=portfolio,
                              action_scheme=action,
                              reward_scheme=reward,
                              feed=feed)


def test_simulated_dynamic_simple_env():
    assert make_env('simulated', 'dynamic', 'simple')


def test_simulated_managed_risk_simple_env():
    assert make_env('simulated', 'managed-risk', 'simple')


@pytest.mark.skip(reason="Authentication Error")
def test_ccxt_dynamic_simple_env():
    for exchange_id in ['coinbasepro', 'coinbase', 'binance', 'bitstamp']:
        assert make_env(exchange_id, 'dynamic', 'simple')


@pytest.mark.skip(reason="Authentication Error")
def test_ccxt_managed_risk_simple_env():
    for exchange_id in ['coinbasepro', 'coinbase', 'binance', 'bitstamp']:
        assert make_env(exchange_id, 'managed-risk', 'simple')


def test_stochastic_dynamic_simple_env():
    assert make_env('stochastic', 'dynamic', 'simple')


def test_stochastic_managed_risk_simple_env():
    assert make_env('stochastic', 'managed-risk', 'simple')


@pytest.mark.skip(reason="GAN exchange is not fully implemented yet.")
def test_gan_dynamic_simple_env():
    assert make_env('gan', 'dynamic', 'simple')


@pytest.mark.skip(reason="GAN exchange is not fully implemented yet.")
def test_gan_managed_risk_simple_env():
    assert make_env('gan', 'managed-risk', 'simple')
