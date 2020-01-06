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


from numbers import Number

from tensortrade.base.exceptions import *
from tensortrade.instruments.quantity import Quantity


class Price:

    def __init__(self, cost, pair: 'TradingPair'):
        self._pair = pair

        if cost < 0:
            raise InvalidNegativeQuantity(cost)
        self.cost = cost

    @property
    def cost(self) -> float:
        """
        Gets the cost of purchasing 1 unit of the quote instrument in terms of the base
        instrument.

        Returns:
            cost : float
                The cost of exchanging the trading pair in terms of the
                base instrument.
        """
        return self._cost

    @cost.setter
    def cost(self, cost: float):
        self._cost = round(cost, self.pair.base.precision)

    @property
    def pair(self) -> 'TradingPair':
        """
        Gets the trading pair associated with cost being stated.

        Returns:
             pair: TradingPair
                The trading pair associated with the cost.
        """
        return self._pair

    def __add__(self, other):
        """Adds two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the right argument of the
                addition operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
        """
        price = None

        if isinstance(other, Price):
            cost = self.cost + other.cost
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(cost, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.cost + other, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __radd__(self, other):
        """Adds two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the left argument of the
                addition operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
        """
        price = None

        if isinstance(other, Price):
            cost = other.cost + self.cost
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(cost, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(other + self.cost, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __iadd__(self, other):
        """Iteratively add to a Price object.

        Arguments:
            other: Union[Price, float, int]
                The object that is the left argument of the
                addition operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """

        if isinstance(other, Price):
            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            self.cost += other.cost

        elif isinstance(other, float) or isinstance(other, int):
            self.cost += other

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return self

    def __sub__(self, other):
        """Subtracts two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the right argument of the
                addition operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """
        price = None

        if isinstance(other, Price):
            cost = self.cost - other.cost
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(cost, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.cost - other, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __rsub__(self, other):
        """Subtracts two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the left argument of the
                addition operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """
        price = None

        if isinstance(other, Price):
            cost = other.cost - self.cost
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(cost, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(other - self.cost, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __mul__(self, other):
        """Multiplies two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the right argument of the
                addition operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """
        price = None

        if isinstance(other, Price):
            cost = self.cost * other.cost
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(cost, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.cost * other, self.pair)

        elif isinstance(other, Quantity):

            if other.instrument != self.pair.quote:
                raise IncompatiblePriceQuantityOperation(self.pair.quote, other.instrument)

            return Quantity(self.pair.base, self.cost * other.size, other.path_id)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __str__(self):
        s = "{0:." + str(self.pair.base.precision) + "f}" + " {1}"
        s = s.format(self.cost, self.pair)
        return s
