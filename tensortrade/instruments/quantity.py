import operator
import warnings

from typing import Union
from numbers import Number

from tensortrade.base.exceptions import InvalidNegativeQuantity, IncompatibleInstrumentOperation, \
    InvalidNonNumericQuantity, QuantityOpPathMismatch


class Quantity:
    """An size of a financial instrument for use in trading."""

    def __init__(self, instrument: 'Instrument', size: float = 0, path_id: str = None):
        if size < 0:
            raise InvalidNegativeQuantity(size)

        self._size = round(size, instrument.precision)
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

    @staticmethod
    def _bool_operation(left: Union['Quantity', float, int],
                        right: Union['Quantity', float, int],
                        bool_op: operator) -> bool:
        right_size = right

        if isinstance(right, Quantity):
            if left.instrument != right.instrument:
                raise IncompatibleInstrumentOperation(left, right)

            right_size = right.size

        if not isinstance(right_size, Number):
            raise InvalidNonNumericQuantity(right_size)

        boolean = bool_op(left.size, right_size)

        if not isinstance(boolean, bool):
            raise Exception("`bool_op` cannot return a non-bool type ({}).".format(boolean))

        return boolean

    @staticmethod
    def _math_operation(left: Union['Quantity', float, int],
                        right: Union['Quantity', float, int],
                        op: operator) -> 'Quantity':
        right_size = right

        if isinstance(right, Quantity):
            if left.instrument != right.instrument:
                raise IncompatibleInstrumentOperation(left, right)

            if left.path_id and right.path_id:
                if left._path_id != right._path_id:
                    raise QuantityOpPathMismatch(left.path_id, right.path_id)

            right_size = right.size

        if not isinstance(right_size, Number):
            raise InvalidNonNumericQuantity(right_size)

        size = op(left.size, right_size)

        return Quantity(instrument=left.instrument, size=size, path_id=left.path_id)

    def __add__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.add)

    def __sub__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.sub)

    def __iadd__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.iadd)

    def __isub__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.isub)

    def __mul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.mul)

    def __truediv__(self, other: Union['Quantity', float, int]) -> 'Quantity':
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
