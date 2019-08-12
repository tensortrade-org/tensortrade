import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List
from timeserio.pipeline import Pipeline


class TradingAgent(object, metaclass=ABCMeta):
    '''Abstract base class for reinforcement learning agents.'''

    def __init__(self):
        pass

    @abstractmethod
    def train(self, steps: int, callback: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        '''Trains the agent.

        # Arguments:

        # Returns:
          pd.DataFrame
        '''
        pass

    @abstractmethod
    def evaluate(self, steps: int, callback: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, observation: List) -> Union[int, tuple]:
        pass

    @abstractmethod
    def add_feature_pipeline(self, pipeline: Pipeline):
        pass
