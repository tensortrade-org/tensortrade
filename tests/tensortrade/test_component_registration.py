
import pytest
import warnings
import ccxt
import tensortrade.actions as actions
import tensortrade.exchanges as exchanges
import tensortrade.rewards as rewards
import tensortrade.slippage as slippage
import tensortrade.environments as envs

from tensortrade.actions import *
from tensortrade.exchanges.simulated import *
from tensortrade.exchanges.live import *
from tensortrade.rewards import *
from tensortrade.slippage import *
from tensortrade.environments import TradingEnvironment

warnings.filterwarnings("ignore")


def test_continuous_action_strategy():
    assert isinstance(actions.get('continuous'), ContinuousActionStrategy)


def test_discrete_action_strategy():
    assert isinstance(actions.get('continuous'), ContinuousActionStrategy)


def test_simulated_exchange():
    assert isinstance(exchanges.get('simulated'), SimulatedExchange)


def test_ccxt_exchanges():
    for exchange_id in ['coinbasepro', 'coinbase', 'binance', 'bitstamp']:
        assert isinstance(exchanges.get(exchange_id), CCXTExchange)


def test_fbm_exchange():
    assert isinstance(exchanges.get('fbm'), FBMExchange)


@pytest.mark.skip(reason="GAN exchange is not fully implemented yet.")
def test_gan_exchange():
    assert isinstance(exchanges.get('gan'), GANExchange)


def test_simple_reward_strategy():
    assert isinstance(rewards.get('simple'), SimpleProfitStrategy)


def test_random_uniform_slippage_model():
    assert isinstance(slippage.get('uniform'), RandomUniformSlippageModel)


def test_basic_environment():
    assert isinstance(envs.get('basic'), TradingEnvironment)


def make_env(exchange: str, action: str, reward: str):
    return TradingEnvironment(exchange=exchange, action_strategy=action, reward_strategy=reward)


def test_simulated_continuous_simple_env():
    assert make_env('simulated', 'continuous', 'simple')


def test_simulated_discrete_simple_env():
    assert make_env('simulated', 'discrete', 'simple')


@pytest.mark.skip(reason="Authentication Error")
def test_ccxt_continuous_simple_env():
    for exchange_id in ['coinbasepro', 'coinbase', 'binance', 'bitstamp']:
        assert make_env(exchange_id, 'continuous', 'simple')


@pytest.mark.skip(reason="Authentication Error")
def test_ccxt_discrete_simple_env():
    for exchange_id in ['coinbasepro', 'coinbase', 'binance', 'bitstamp']:
        assert make_env(exchange_id, 'continuous', 'simple')


def test_fbm_continuous_simple_env():
    assert make_env('fbm', 'continuous', 'simple')


def test_fbm_discrete_simple_env():
    assert make_env('fbm', 'discrete', 'simple')


@pytest.mark.skip(reason="GAN exchange is not fully implemented yet.")
def test_gan_continuous_simple_env():
    assert make_env('gan', 'continuous', 'simple')


@pytest.mark.skip(reason="GAN exchange is not fully implemented yet.")
def test_gan_discrete_simple_env():
    assert make_env('gan', 'discrete', 'simple')
