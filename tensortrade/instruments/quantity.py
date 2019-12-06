import operator


class Quantity:
    """An amount of a financial instrument for use in trading."""

    def __init__(self,
                 amount: float,
                 instrument: 'Instrument',
                 order_id: str = None,
                 wallet_id: str = None):

        if amount < 0:
            raise Exception("Invalid Quantity. Amounts cannot be negative.")

        self._amount = round(amount, instrument.precision)
        self._instrument = instrument
        self._order_id = None
        self._wallet_id = None

    @property
    def amount(self) -> float:
        return self._amount

    @amount.setter
    def amount(self, amount: float):
        self._amount = amount

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: 'Exchange'):
        raise ValueError("You cannot change a Quantity's Instrument after initialization.")

    @property
    def order_id(self) -> str:
        return self._order_id

    @order_id.setter
    def order_id(self, order_id: str):
        self._order_id = order_id

    @property
    def wallet_id(self) -> str:
        return self._wallet_id

    @wallet_id.setter
    def wallet_id(self, wallet_id: str):
        self._wallet_id = wallet_id

    def is_locked(self) -> bool:
        return bool(self._order_id)

    @staticmethod
    def _quantity_op(left, right, op):
        right_amount = right

        if isinstance(right, Quantity):
            if left.instrument != right.instrument:
                raise Exception("Instruments are not of the same type.")

            right_amount = right.amount

        if not str(right_amount).isnumeric():
            raise Exception("Can't perform operation with non-numeric quantity.")

        amount = round(op(left.amount, right_amount), left.instrument.precision)

        if amount < 0:
            raise Exception("Quantities must be non-negative.")

        return Quantity(amount, left.instrument)

    def __add__(self, other):
        return Quantity._quantity_op(self, other, operator.add)

    def __sub__(self, other):
        return Quantity._quantity_op(self, other, operator.sub)

    def __mul__(self, other):
        return Quantity._quantity_op(self, other, operator.add)

    def __truediv__(self, other):
        return Quantity._quantity_op(self, other, operator.truediv)

    def __str__(self):
        s = "{0:." + str(self.instrument.precision) + "f}" + " {1}"
        s = s.format(self.amount, self.instrument.symbol)
        return s

    def __repr__(self):
        return str(self)
