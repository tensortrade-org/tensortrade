
import pytest

from tensortrade.actions import DiscreteActionStrategy
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.environments import TradingEnvironment

from tensortrade.strategies import TradingStrategy


class ConcreteTradingStrategy(TradingStrategy):

    def __init__(self, environment: 'TradingEnvironment'):
        super(ConcreteTradingStrategy, self).__init__(environment)

    def restore_agent(self, path: str):
        pass

    def save_agent(self, path: str):
        pass

    def tune(self, steps_per_train: int, steps_per_test: int, episode_callback=None) -> pd.DataFrame:
        pass

    def run(self, steps: int = None, episodes: int = None, testing: bool = False,
            episode_callback=None) -> pd.DataFrame:
        pass


config = {
    'base_instrument': 'USD',
    'products': 'ETH',
    'credentials': {
        'api_key': '48hg34wydghi7ef',
        'api_secret_key': '0984hgoe8d7htg'
    }
}


def test_injects_trading_strategy_with_context():

    with TradingStrategy(**config) as tc:

        strategy = TradingEnvironment(
            exchange='simulated',
            action_strategy='discrete',
            reward_strategy='simple'
        )

        assert strategy.reward_strategy.context == tc
        assert strategy.action_strategy.context == tc
        assert strategy.exchange.context == tc
        assert strategy.context == tc


@pytest.skip(msg="ModuleNotFoundError: No module named \'tensorflow.contrib\'")
def test_injects_string_initialized_strategy_with_context():

    with TradingStrategy(**config) as tc:

        strategy = TradingEnvironment(
            exchange='fbm',
            action_strategy='discrete',
            reward_strategy='simple'
        )

        assert strategy.reward_strategy.context == tc
        assert strategy.action_strategy.context == tc
        assert strategy.exchange.context == tc
        assert strategy.context == tc


