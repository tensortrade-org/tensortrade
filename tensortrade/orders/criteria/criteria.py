
from abc import abstractmethod, ABCMeta
from typing import List, Union


class Criteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    @abstractmethod
    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        raise NotImplementedError()
