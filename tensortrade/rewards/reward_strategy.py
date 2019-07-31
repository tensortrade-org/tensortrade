import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List, Dict, Callable

from tensortrade.exchanges import AssetExchange


class RewardStrategy(object, metaclass=ABCMeta):
    dtype: type = np.float16

    @abstractmethod
    def __init__(self):
        pass

    def set_dtype(self, dtype):
        self.dtype = dtype

    def reset(self):
        pass

    @abstractmethod
    def reward(self, current_step: int, exchange: AssetExchange) -> float:
        raise NotImplementedError()
