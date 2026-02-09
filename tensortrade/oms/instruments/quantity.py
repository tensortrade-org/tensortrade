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
from __future__ import annotations

import operator
from collections.abc import Callable
from decimal import ROUND_DOWN, Decimal
from functools import total_ordering
from numbers import Number
from typing import TYPE_CHECKING, Any, TypeVar

from tensortrade.core.exceptions import (
    IncompatibleInstrumentOperation,
    InvalidNegativeQuantity,
    InvalidNonNumericQuantity,
    QuantityOpPathMismatch,
)
from tensortrade.oms.instruments.exchange_pair import ExchangePair

if TYPE_CHECKING:
    from tensortrade.oms.instruments.instrument import Instrument


T = TypeVar("T")


@total_ordering
class Quantity:
    """A size of a financial instrument for use in trading.

    Parameters
    ----------
    instrument : `Instrument`
        The unit of the quantity.
    size : `Decimal`
        The number of units of the instrument.
    path_id : str, optional
        The path order_id that this quantity is allocated for and associated
        with.

    Raises
    ------
    InvalidNegativeQuantity
        Raised if the `size` of the quantity being created is negative.
    """

    def __init__(
        self,
        instrument: Instrument,
        size: Decimal | float | int,
        path_id: str | None = None,
    ):
        size = Decimal(size)
        if size < 0:
            if abs(size) > Decimal(10) ** (-instrument.precision):
                raise InvalidNegativeQuantity(size)
            else:
                size = Decimal(0)

        self.instrument = instrument
        self.size = size
        self.path_id = path_id

    @property
    def is_locked(self) -> bool:
        """If quantity is locked for an order. (bool, read-only)"""
        return bool(self.path_id)

    def lock_for(self, path_id: str) -> Quantity:
        """Locks a quantity for an `Order` identified associated with `path_id`.

        Parameters
        ----------
        path_id : str
            The identification of the order path.

        Returns
        -------
        `Quantity`
            A locked quantity for an order path.
        """
        return Quantity(self.instrument, self.size, path_id)

    def convert(self, exchange_pair: ExchangePair) -> Quantity:
        """Converts the quantity into the value of another instrument based
        on its exchange rate from an exchange.

        Parameters
        ----------
        exchange_pair : `ExchangePair`
            The exchange pair to use for getting the quoted price to perform
            the conversion.

        Returns
        -------
        `Quantity`
            The value of the current quantity in terms of the quote instrument.
        """
        if self.instrument == exchange_pair.pair.base:
            instrument = exchange_pair.pair.quote
            converted_size = self.size / exchange_pair.price
        else:
            instrument = exchange_pair.pair.base
            converted_size = self.size * exchange_pair.price
        return Quantity(instrument, converted_size, self.path_id)

    def free(self) -> Quantity:
        """Gets the free version of this quantity.

        Returns
        -------
        `Quantity`
            The free version of the quantity.
        """
        return Quantity(self.instrument, self.size)

    def quantize(self) -> Quantity:
        """Computes the quantization of current quantity in terms of the instrument's
        precision.

        Returns
        -------
        `Quantity`
            The quantized quantity.
        """
        return Quantity(
            self.instrument,
            self.size.quantize(Decimal(10) ** -self.instrument.precision),
            self.path_id,
        )

    def as_float(self) -> float:
        """Gets the size as a `float`.

        Returns
        -------
        float
            The size as a floating point number.
        """
        return float(self.size)

    def contain(self, exchange_pair: ExchangePair):
        """Contains the size of the quantity to be compatible with the settings
        of a given exchange.

        Parameters
        ----------
        exchange_pair : `ExchangePair`
            The exchange pair containing the exchange the quantity must be
            compatible with.

        Returns
        -------
        `Quantity`
            A quantity compatible with the given exchange.
        """
        options = exchange_pair.exchange.options
        price = exchange_pair.price

        if exchange_pair.pair.base == self.instrument:
            size = self.size
            return Quantity(
                self.instrument, min(size, options.max_trade_size), self.path_id
            )

        size = self.size * price
        if size < options.max_trade_size:
            return Quantity(self.instrument, self.size, self.path_id)

        max_trade_size = Decimal(options.max_trade_size)
        contained_size = max_trade_size / price
        contained_size = contained_size.quantize(
            Decimal(10) ** -self.instrument.precision, rounding=ROUND_DOWN
        )
        return Quantity(self.instrument, contained_size, self.path_id)

    @staticmethod
    def validate(
        left: Quantity | float | int, right: Quantity | float | int
    ) -> tuple[Quantity, Quantity]:
        """Validates the given left and right arguments of a numeric or boolean
        operation.

        Parameters
        ----------
        left : Quantity | Number
            The left argument of an operation.
        right : Quantity | Number
            The right argument of an operation.

        Returns
        -------
        `Tuple[Quantity, Quantity]`
            The validated quantity arguments to use in a numeric or boolean
            operation.

        Raises
        ------
        IncompatibleInstrumentOperation
            Raised if the instruments left and right quantities are not equal.
        QuantityOpPathMismatch
            Raised if
                - One argument is locked and the other argument is not.
                - Both arguments are locked quantities with unequal path_ids.
        InvalidNonNumericQuantity
            Raised if either argument is a non-numeric object.
        Exception
            If the operation is not valid.
        """
        match (left, right):
            case (Quantity() as l, Quantity() as r):
                if l.instrument != r.instrument:
                    raise IncompatibleInstrumentOperation(l, r)

                if (l.path_id and r.path_id) and (l.path_id != r.path_id):
                    raise QuantityOpPathMismatch(l.path_id, r.path_id)
                elif l.path_id and not r.path_id:
                    r.path_id = l.path_id
                elif not l.path_id and r.path_id:
                    l.path_id = r.path_id

                return l, r

            case (Number() as size, Quantity() as r):
                return Quantity(r.instrument, size, r.path_id), r

            case (Quantity() as l, Number() as size):
                return l, Quantity(l.instrument, size, l.path_id)

            case (Quantity(), _):
                raise InvalidNonNumericQuantity(right)

            case (_, Quantity()):
                raise InvalidNonNumericQuantity(left)

            case _:
                raise Exception(
                    f"Invalid quantity operation arguments: {left} and {right}"
                )

    @staticmethod
    def _bool_op(
        left: Quantity | float | int,
        right: Quantity | float | int,
        op: Callable[[Decimal, Decimal], bool],
    ) -> bool:
        """Performs a generic boolean operation on two quantities.

        Parameters
        ----------
        left : `Union[Quantity, Number]`
            The left argument of the operation.
        right : `Union[Quantity, Number]`
            The right argument of the operation.
        op : `Callable[[T, T], bool]`
            The boolean operation to be used.

        Returns
        -------
        bool
            The result of performing `op` on with `left` and `right`.
        """
        left, right = Quantity.validate(left, right)
        boolean = op(left.size, right.size)
        return boolean

    @staticmethod
    def _math_op(
        left: Quantity | float | int,
        right: Quantity | float | int,
        op: Callable[[Decimal, Decimal], Decimal],
    ) -> Quantity:
        """Performs a generic numeric operation on two quantities.

        Parameters
        ----------
        left : Quantity | Number
            The left argument of the operation.
        right : Quantity | Number
            The right argument of the operation.
        op : Callable[[Decimal, Decimal], Decimal]
            The numeric operation to be used.

        Returns
        -------
        `Quantity`
            The result of performing `op` on with `left` and `right`.
        """
        left, right = Quantity.validate(left, right)
        size = op(left.size, right.size)
        return Quantity(left.instrument, size, left.path_id)

    def __add__(self, other: Quantity) -> Quantity:
        return Quantity._math_op(self, other, operator.add)

    def __sub__(self, other: Quantity | float | int) -> Quantity:
        return Quantity._math_op(self, other, operator.sub)

    def __iadd__(self, other: Quantity | float | int) -> Quantity:
        return Quantity._math_op(self, other, operator.iadd)

    def __isub__(self, other: Quantity | float | int) -> Quantity:
        return Quantity._math_op(self, other, operator.isub)

    def __mul__(self, other: Quantity | float | int) -> Quantity:
        return Quantity._math_op(self, other, operator.mul)

    def __rmul__(self, other: Quantity | float | int) -> Quantity:
        return Quantity.__mul__(self, other)

    def __lt__(self, other: Quantity | float | int) -> bool:
        return Quantity._bool_op(self, other, operator.lt)

    def __eq__(self, other: Any) -> bool:
        return Quantity._bool_op(self, other, operator.eq)

    def __ne__(self, other: Any) -> bool:
        return Quantity._bool_op(self, other, operator.ne)

    def __neg__(self) -> bool:
        return operator.neg(self.size)

    def __str__(self) -> str:
        s = "{0:." + str(self.instrument.precision) + "f}" + " {1}"
        s = s.format(self.size, self.instrument.symbol)
        return s

    def __repr__(self) -> str:
        return str(self)
