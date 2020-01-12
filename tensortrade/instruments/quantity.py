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

from typing import Union, Tuple
from numbers import Number

from tensortrade.base.exceptions import InvalidNegativeQuantity, IncompatibleInstrumentOperation, \
    InvalidNonNumericQuantity, QuantityOpPathMismatch, IncompatiblePriceQuantityOperation, \
    IncompatibleTradingPairOperation


class Quantity:
    """An size of a financial instrument for use in trading.
    """

    def __init__(self, instrument: 'Instrument', size: float = 0, path_id: str = None):
        if size < 0:
            raise InvalidNegativeQuantity(size)

        self._size = size
        self._instrument = instrument
        self._path_id = path_id

    @property
    def size(self) -> float:
        return self._size

    @size.setter
    def size(self, size: float):
        self._size = size

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: 'Exchange'):
        raise ValueError("You cannot change a Quantity's Instrument after initialization.")

    @property
    def path_id(self) -> str:
        return self._path_id

    @path_id.setter
    def path_id(self, path_id: str):
        self._path_id = path_id

    @property
    def is_locked(self) -> bool:
        return bool(self._path_id)

    def lock_for(self, path_id: str):
        self._path_id = path_id

    def free(self):
        return Quantity(self.instrument, self.size)

    @staticmethod
    def validate(left, right) -> Tuple['Quantity', 'Quantity']:

        if isinstance(left, Quantity) and isinstance(right, Quantity):
            if left.instrument != right.instrument:
                raise IncompatibleInstrumentOperation(left, right)

            if (left.path_id and right.path_id) and (left.path_id != right.path_id):
                raise QuantityOpPathMismatch(left.path_id, right.path_id)

            elif left.path_id and not right.path_id:
                right.path_id = left.path_id

            elif not left.path_id and right.path_id:
                left.path_id = right.path_id

            return left, right

        elif isinstance(left, Number) and isinstance(right, Quantity):
            left = Quantity(right.instrument, float(left), right.path_id)
            return left, right

        elif isinstance(left, Quantity) and isinstance(right, Number):
            right = Quantity(left.instrument, float(right), left.path_id)
            return left, right

        elif isinstance(left, Quantity):
            raise InvalidNonNumericQuantity(right)

        elif isinstance(right, Quantity):
            raise InvalidNonNumericQuantity(left)

        return left, right

    @staticmethod
    def _bool_operation(left: Union['Quantity', float, int],
                        right: Union['Quantity', float, int],
                        bool_op: operator) -> bool:
        left, right = Quantity.validate(left, right)

        boolean = bool_op(left.size, right.size)

        if not isinstance(boolean, bool):
            raise Exception("`bool_op` cannot return a non-bool type ({}).".format(boolean))

        return boolean

    @staticmethod
    def _math_operation(left: Union['Quantity', float, int],
                        right: Union['Quantity', float, int],
                        op: operator) -> 'Quantity':
        left, right = Quantity.validate(left, right)

        size = op(left._size, right._size)
        return Quantity(left.instrument, size, left.path_id)

    def __add__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.add)

    def __sub__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.sub)

    def __iadd__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.iadd)

    def __isub__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.isub)

    def __mul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        if isinstance(other, Price):
            if other.pair.quote != self.instrument:
                raise IncompatiblePriceQuantityOperation(other.pair.quote, self.instrument)
            return Quantity(other.pair.base, other.rate * self.size, self.path_id)
        return Quantity._math_operation(self, other, operator.mul)

    def __rmul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity.__mul__(other, self)

    def __truediv__(self, other: Union['Quantity', 'Instrument', float, int]) -> Union['Quantity', 'Price']:
        if isinstance(other, Price):
            if other.pair.base != self.instrument:
                raise IncompatiblePriceQuantityOperation(other.pair.quote, self.instrument)
            return Quantity(other.pair.quote, self.size / other.rate, self.path_id)
        elif other.__class__.__name__ == "Instrument":
            return Price(self.size, self.instrument / other)

        elif isinstance(other, Quantity):
            if self.instrument != other.instrument:
                return Price(self.size / other.size, self.instrument / other.instrument)

        return Quantity._math_operation(self, other, operator.truediv)

    def __lt__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.lt)

    def __gt__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.gt)

    def __eq__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.eq)

    def __ne__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.ne)

    def __neg__(self) -> bool:
        return operator.neg(self.size)

    def __str__(self):
        s = "{0:." + str(self.instrument.precision) + "f}" + " {1}"
        s = s.format(self.size, self.instrument.symbol)
        return s

    def __repr__(self):
        return str(self)


class Price:

    def __init__(self, rate, pair: 'TradingPair'):
        self._pair = pair

        if rate < 0:
            raise InvalidNegativeQuantity(rate)
        self.rate = rate

    @property
    def rate(self) -> float:
        """
        Gets the rate of purchasing 1 unit of the quote instrument in terms of the base
        instrument.

        Returns:
            rate : float
                The rate of exchanging the trading pair in terms of the
                base instrument.
        """
        return self._rate

    @rate.setter
    def rate(self, rate: float):
        self._rate = round(rate, self.pair.base.precision)

    @property
    def pair(self) -> 'TradingPair':
        """
        Gets the trading pair associated with rate being stated.

        Returns:
             pair: TradingPair
                The trading pair associated with the rate.
        """
        return self._pair

    def __add__(self, other: Union['Price', float, int]):
        """Adds two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the right argument of the
                operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
        """
        price = None

        if isinstance(other, Price):
            rate = self.rate + other.rate
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(rate, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.rate + other, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __radd__(self, other: Union['Price', float, int]):
        """Adds two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the left argument of the
                operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
        """
        price = None

        if isinstance(other, Price):
            rate = other.rate + self.rate
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(rate, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(other + self.rate, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __iadd__(self, other: Union['Price', float, int]):
        """Iteratively add to a Price object.

        Arguments:
            other: Union[Price, float, int]
                The object that is the left argument of the
                operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """

        if isinstance(other, Price):
            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            self.rate += other.rate

        elif isinstance(other, float) or isinstance(other, int):
            self.rate += other

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return self

    def __sub__(self, other: Union['Price', float, int]):
        """Subtracts two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the right argument of the
                operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """
        price = None

        if isinstance(other, Price):
            rate = self.rate - other.rate
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(rate, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.rate - other, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __rsub__(self, other: Union['Price', float, int]):
        """Subtracts two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the left argument of the
                operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """
        price = None

        if isinstance(other, Price):
            rate = other.rate - self.rate
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(rate, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(other - self.rate, self.pair)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __mul__(self, other: Union['Price', 'Quantity', float, int]):
        """Multiplies two PriceType objects together.

        Arguments:
            other: Union[Price, float, int]
                The object that is the right argument of the
                operation.

        Raises:
            IncompatibleTradingPairException:
                If (other : Price) and the pairs do not equal each other.
            InvalidNonNumericQuantity:
                If not (other : Union[Price, Number]).
        """
        price = None

        if isinstance(other, Price):
            rate = self.rate * other.rate
            pair = self.pair or other.pair

            if self.pair != other.pair:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            price = Price(rate, pair)

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.rate * other, self.pair)

        elif isinstance(other, Quantity):

            if other.instrument != self.pair.quote:
                raise IncompatiblePriceQuantityOperation(self.pair.quote, other.instrument)

            return Quantity(self.pair.base, self.rate * other.size, other.path_id)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __rmul__(self, other: Union['Price', 'Quantity', float, int]):
        """Multiplies two PriceType objects together.

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
        return self.__mul__(other)

    def __truediv__(self, other: Union['Price', float, int]):
        """Divides two PriceType objects together.

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

            if self.pair.base == other.pair.base and self.pair.quote != other.pair.quote:
                return Price(self.rate / other.rate, other.pair.quote / self.pair.quote)

            elif self.pair.base != other.pair.base and self.pair.quote == other.pair.quote:
                return Price(self.rate / other.rate, self.pair.base / other.pair.base)

            elif self.pair.base != other.pair.base and self.pair.quote != other.pair.quote:
                raise IncompatibleTradingPairOperation(self.pair, other.pair)

            return self.rate / other.rate

        elif isinstance(other, float) or isinstance(other, int):
            price = Price(self.rate / other, self.pair)

        elif isinstance(other, Quantity):
            raise IncompatiblePriceQuantityOperation(self, other)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __rtruediv__(self, other: Union['Price', float, int]):
        """Divides two PriceType objects together.

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

        if isinstance(other, float) or isinstance(other, int):
            price = Price(other / self.rate, self.pair.quote / self.pair.base)

        elif isinstance(other, Quantity):
            raise IncompatiblePriceQuantityOperation(self, other)

        elif not isinstance(other, Number):
            raise InvalidNonNumericQuantity(other)

        return price

    def __str__(self):
        s = "{0:." + str(self.pair.base.precision) + "f}" + " {1}"
        s = s.format(self.rate, self.pair)
        return s

    def __repr__(self):
        return str(self)
