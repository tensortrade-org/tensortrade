from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List

from tensortrade.agents.features import FeaturePipeline
from tensortrade.exchanges.performance import TradingPerformance


class TradingAgent(object, metaclass=ABCMeta):
    '''Abstract base class for reinforcement learning agents.'''

    def __init__(self):
        pass

    @abstractmethod
    def train(self, steps: int, callback: Callable[[TradingPerformance], bool]) -> TradingPerformance:
        '''Trains the agent.

        # Arguments:
          steps:
          callback: A `Callable`, object of type TradingPerformance, which

        # Returns:
          TradingPerformance
        '''
        pass

    @abstractmethod
    def evaluate(self, steps: int, callback: Callable[[TradingPerformance], bool]) -> TradingPerformance:
        pass

    @abstractmethod
    def predict(self, observation: List) -> Union[int, tuple]:
        pass

    @abstractmethod
    def add_feature_pipeline(self, pipeline: FeaturePipeline):
        pass
