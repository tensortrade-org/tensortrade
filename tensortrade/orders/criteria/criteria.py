
from abc import abstractmethod, ABCMeta
from typing import List, Union, Callable


CriteriaType = Callable[['Order', 'Exchange'], bool]


class Criteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    @abstractmethod
    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        raise NotImplementedError

    def __and__(self, other: CriteriaType) -> 'Criteria':
        return And(self, other)

    def __or__(self, other: CriteriaType) -> 'Criteria':
        return Or(self, other)


class And(Criteria):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        self._left = left
        self._right = right

    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        return self._left(order, exchange) and self._right(order, exchange)


class Or(Criteria):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        self._left = left
        self._right = right

    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        return self._left(order, exchange) or self._right(order, exchange)
