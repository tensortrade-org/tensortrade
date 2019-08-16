import pandas as pd

from abc import ABCMeta, abstractmethod

from tensortrade.exchanges import AssetExchange


class RewardStrategy(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    def set_dtype(self, dtype):
        self.dtype = dtype

    def reset(self):
        pass

    @abstractmethod
    def get_reward(self, current_step: int, exchange: AssetExchange) -> float:
        raise NotImplementedError()
