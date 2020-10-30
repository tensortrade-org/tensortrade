# Copyright 2020 The TensorTrade Authors.
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
from typing import Callable, Union, TypeVar
from enum import Enum

from tensortrade.oms.orders import TradeSide, Order
from tensortrade.oms.exchanges import Exchange


class Criteria(object, metaclass=ABCMeta):
    """A criteria to be satisfied before an order will be executed."""

    @abstractmethod
    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        """Checks whether the `order` is executable on `exchange`.

        Parameters
        ----------
        order : `Order`
            An order.
        exchange : `Exchange`
            The exchange to check.

        Returns
        -------
        bool
            Whether `order` is executable on `exchange`.
        """
        raise NotImplementedError

    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradable(order.pair):
            return False
        return self.check(order, exchange)

    def __and__(self, other: 'Callable[[Order, Exchange], bool]') -> 'Criteria':
        return CriteriaBinOp(self, other, operator.and_, "&")

    def __or__(self, other: 'Callable[[Order, Exchange], bool]') -> 'Criteria':
        return CriteriaBinOp(self, other, operator.or_, "|")

    def __xor__(self, other: 'Callable[[Order, Exchange], bool]') -> 'Criteria':
        return CriteriaBinOp(self, other, operator.xor, "^")

    def __invert__(self):
        return NotCriteria(self)

    def __repr__(self):
        return str(self)


class CriteriaBinOp(Criteria):
    """A class for using a binary operation for criteria.

    Parameters
    ----------
    left : `Callable[[Order, Exchange], bool]`
        The left criteria argument.
    right : `Callable[[Order, Exchange], bool]`
        The right criteria argument.
    op : `Callable[[bool, bool], bool]`
        The binary boolean operation.
    op_str : str
        The string representing the op.
    """

    def __init__(self,
                 left: 'Callable[[Order, Exchange], bool]',
                 right: 'Callable[[Order, Exchange], bool]',
                 op: Callable[[bool, bool], bool],
                 op_str: str):
        self.left = left
        self.right = right
        self.op = op
        self.op_str = op_str

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        left = self.left(order, exchange)
        right = self.right(order, exchange)

        return self.op(left, right)

    def __str__(self) -> str:
        is_left_op = isinstance(self.left, CriteriaBinOp)
        is_right_op = isinstance(self.right, CriteriaBinOp)

        if is_left_op and is_right_op:
            return "({}) {} ({})".format(self.left, self.op_str, self.right)
        elif is_left_op and not is_right_op:
            return "({}) {} {}".format(self.left, self.op_str, self.right)
        elif not is_left_op and is_right_op:
            return "{} {} ({})".format(self.left, self.op_str, self.right)

        return "{} {} {}".format(self.left, self.op_str, self.right)


class NotCriteria(Criteria):
    """A criteria to invert the truth value of another criteria.

    Parameters
    ----------
    criteria : `Callable[[Order, Exchange], bool]`
        The criteria to invert the truth value of.
    """

    def __init__(self,
                 criteria: 'Callable[[Order, Exchange], bool]') -> None:
        self.criteria = criteria

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        return not self.criteria(order, exchange)

    def __str__(self) -> str:
        if isinstance(self.criteria, CriteriaBinOp):
            return f"~({self.criteria})"
        return f"~{self.criteria}"


class Limit(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price, hidden from the public
    order book.

    Parameters
    ----------
    limit_price : float
        The quote price to check for execution.
    """

    def __init__(self, limit_price: float) -> None:
        self.limit_price = limit_price

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        price = exchange.quote_price(order.pair)

        buy_satisfied = (order.side == TradeSide.BUY and price <= self.limit_price)
        sell_satisfied = (order.side == TradeSide.SELL and price >= self.limit_price)

        return buy_satisfied or sell_satisfied

    def __str__(self) -> str:
        return f"<Limit: price={self.limit_price}>"


class StopDirection(Enum):
    """An enumeration for the directions of a stop criteria."""

    UP = "up"
    DOWN = "down"

    def __str__(self) -> str:
        return str(self.value)


class Stop(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is above or below a specific price.

    Parameters
    ----------
    direction : `Union[StopDirection, str]`
        The direction to watch for the stop criteria.
    percent : float
        The percentage of the current price to use for watching.
    """

    def __init__(self,
                 direction: 'Union[StopDirection, str]',
                 percent: float) -> None:
        self.direction = StopDirection(direction)
        self.percent = percent

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        price = exchange.quote_price(order.pair)
        percent = abs(price - order.price) / order.price

        is_take_profit = (self.direction == StopDirection.UP) and (price >= order.price)
        is_stop_loss = (self.direction == StopDirection.DOWN) and (price <= order.price)

        return (is_take_profit or is_stop_loss) and percent >= self.percent

    def __str__(self):
        return f"<Stop: direction={self.direction}, percent={self.percent}>"


class Timed(Criteria):
    """An order criteria for waiting a certain amount of time for execution.

    Parameters
    ----------
    duration : float
        The amount of time to wait.
    """

    def __init__(self, duration: float):
        self.duration = duration

    def check(self, order: 'Order', exchange: 'Exchange'):
        return (order.clock.step - order.created_at) <= self.duration

    def __str__(self):
        return f"<Timed: duration={self.duration}>"
