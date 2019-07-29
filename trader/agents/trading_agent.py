from abc import ABCMeta, abstractmethod
from typing import Callable, List

from trader.features import FeaturePipeline
from trader.performance import TradingPerformance


class TradingAgent(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, steps: int, callback: Callable[[TradingPerformance], [bool]]) -> TradingPerformance:
        pass

    @abstractmethod
    def evaluate(self, steps: int, callback: Callable[[TradingPerformance], [bool]]) -> TradingPerformance:
        pass

    @abstractmethod
    def predict(self, observation: List) -> int | tuple:
        pass

    @abstractmethod
    def add_feature_pipeline(self, pipeline: FeaturePipeline):
        pass
