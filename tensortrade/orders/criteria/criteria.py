# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


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
        return AND(self, other)

    def __or__(self, other: CriteriaType) -> 'Criteria':
        return OR(self, other)

    def __xor__(self, other: CriteriaType) -> 'Criteria':
        return XOR(self, other)

    def __invert__(self):
        return NOT(self)

    def __repr__(self):
        return str(self)


class CriteriaBinOp(Criteria):

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
        is_left_op = isinstance(self.left, CriteriaBinOp)
        is_right_op = isinstance(self.right, CriteriaBinOp)
        if is_left_op and is_right_op:
            return "({}) {} ({})".format(self.left, self.op_str, self.right)
        elif is_left_op and not is_right_op:
            return "({}) {} {}".format(self.left, self.op_str, self.right)
        elif not is_left_op and is_right_op:
            return "{} {} ({})".format(self.left, self.op_str, self.right)
        return "{} {} {}".format(self.left, self.op_str, self.right)


class AND(CriteriaBinOp):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        super().__init__(left, right, operator.and_, "&")


class OR(CriteriaBinOp):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        super().__init__(left, right, operator.or_, "|")


class XOR(CriteriaBinOp):

    def __init__(self, left: CriteriaType, right: CriteriaType):
        super().__init__(left, right, operator.xor, "^")


class NOT(Criteria):

    def __init__(self, criteria: CriteriaType):
        self.criteria = criteria

    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        return not self.criteria(order, exchange)

    def __str__(self):
        if isinstance(self.criteria, CriteriaBinOp):
            return "~({})".format(self.criteria)
        return "~{}".format(self.criteria)
