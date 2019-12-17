
import pytest
import pandas as pd

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
