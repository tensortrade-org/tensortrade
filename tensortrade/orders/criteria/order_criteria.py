
from abc import abstractmethod, ABCMeta
from typing import List, Union


class OrderCriteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    @abstractmethod
    def is_satisfied(self, order: 'Order', exchange: 'Exchange') -> bool:
        raise NotImplementedError()
