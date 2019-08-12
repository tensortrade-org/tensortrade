from abc import ABCMeta, abstractmethod
from typing import Tuple

from tensortrade.exchanges import AssetExchange


class SlippageModel(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def fill_order(self, amount: float, price: float, exchange: AssetExchange) -> Tuple[float, float]:
        raise NotImplementedError()
