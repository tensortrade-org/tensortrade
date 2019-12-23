
import operator

from abc import abstractmethod, ABCMeta
from typing import Callable


CriteriaType = Callable[['Order', 'Exchange'], bool]


class Criteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    @abstractmethod
    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        raise NotImplementedError

    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradable(order.pair):
            return False
        return self.call(order, exchange)

    def __and__(self, other: CriteriaType) -> 'Criteria':
        return And(self, other)

    def __or__(self, other: CriteriaType) -> 'Criteria':
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def __repr__(self):
        return str(self)


class CriteriaBinaryOp(Criteria):

    def __init__(self,
                 left: CriteriaType,
                 right: CriteriaType,
                 op: Callable[[bool, bool], bool],
                 op_str: str):
        self.left = left
        self.right = right
        self.op = op
        self.op_str = op_str

    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        left = self.left(order, exchange)
        right = self.right(order, exchange)
        return self.op(left, right)

    def __str__(self):
        is_left_op = isinstance(self.left, CriteriaBinaryOp)
        is_right_op = isinstance(self.right, CriteriaBinaryOp)
        if is_left_op and is_right_op:
            return "({}) {} ({})".format(self.left, self.op_str, self.right)
        elif is_left_op and not is_right_op:
            return "({}) {} {}".format(self.left, self.op_str, self.right)
        elif not is_left_op and is_right_op:
            return "{} {} ({})".format(self.left, self.op_str, self.right)
        return "{} {} {}".format(self.left, self.op_str, self.right)


class And(CriteriaBinaryOp):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        super().__init__(left, right, operator.and_, "&")


class Or(CriteriaBinaryOp):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        super().__init__(left, right, operator.or_, "|")


class Not(Criteria):

    def __init__(self, criteria: CriteriaType):
        self.criteria = criteria

    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        return not self.criteria(order, exchange)

    def __str__(self):
        if isinstance(self.criteria, CriteriaBinaryOp):
            return "~({})".format(self.criteria)
        return "~{}".format(self.criteria)
