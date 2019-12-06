
from abc import abstractmethod, ABCMeta
from typing import List, Union


class OrderCriteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    def __init__(self):
        pass

    @abstractmethod
    def is_executable(self, order: 'Order', exchange: 'Exchange') -> bool:
        raise NotImplementedError()
