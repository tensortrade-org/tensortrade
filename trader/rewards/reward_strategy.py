import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List, Callable


class RewardStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def reset(self):
        pass

    @abstractmethod
    def reward(self,
               current_step: int,
               balance: float,
               net_worth: float,
               assets_held: Dict[str, float],
               trades: pd.DataFrame,
               performance: pd.DataFrame) -> float:
        raise NotImplementedError()
