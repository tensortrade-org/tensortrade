
from abc import abstractmethod, ABCMeta
from typing import List, Union


class OrderCriteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    def __init__(self, enabled_pairs: Union[List['TradingPair'], 'TradingPair'] = None):
        if enabled_pairs is not None:
            self.enabled_pairs = enabled_pairs if isinstance(
                enabled_pairs, list) else [enabled_pairs]
        else:
            self.enabled_pairs = None

    def is_pair_enabled(self, pair: 'TradingPair') -> bool:
        return self.enabled_pairs is None or pair in self.enabled_pairs

    @abstractmethod
    def is_executable(self, pair: 'TradingPair', exchange: 'Exchange') -> bool:
        raise NotImplementedError()
