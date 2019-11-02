import pytest
import pandas as pd

from tensortrade.actions import DiscreteActionStrategy
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade import TradingContext
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
    'exchanges': {
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
    },
    'actions': {
        'n_actions': 24
    },
    'rewards': {
        'amount': 100
    },
    'features': {
        'shape': (50, 70)
    }
}


def test_injects_trading_strategy_with_context():

    with TradingContext(**config) as tc:

        env = TradingEnvironment(
            exchange='simulated',
            action_strategy='discrete',
            reward_strategy='simple'
        )

        strategy = ConcreteTradingStrategy(environment=env)

        assert hasattr(strategy.environment.exchange.context, 'credentials')
        assert strategy.environment.exchange.context.credentials == config['exchanges']['credentials']

        assert hasattr(strategy.environment.action_strategy.context, 'n_actions')
        assert strategy.environment.action_strategy.context.n_actions == 24

        print(strategy.environment.reward_strategy.context.data)
        assert hasattr(strategy.environment.reward_strategy.context, 'amount')
        assert strategy.environment.reward_strategy.context.amount == 100
