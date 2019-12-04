class Quantity:
    """An amount of a financial instrument for use trading."""

    def __init__(self, amount: float, instrument: 'Instrument'):
        self._amount = round(amount, instrument.precision)
        self._instrument = instrument

    @property
    def amount(self) -> float:
        return self._amount

    @amount.setter
    def amount(self, amount: float):
        self._amount = amount

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    def __add__(self, other):
        if isinstance(other, Quantity):
            if self.instrument != other.instrument:
                raise Exception("Instruments are not of the same type.")

            return Quantity(self.amount + other.amount, self.instrument)

        if not str(other).isnumeric():
            raise Exception("Can't add with non-numeric quantity.")

        other = float(other)

        return Quantity(self.amount + other, self.instrument)

    def __mul__(self, other):
        if not str(other).isnumeric():
            raise Exception("Can't multiply with non-numeric quantity.")

        other = float(other)

        return Quantity(self.amount * other, self.instrument)

    def __str__(self):
        return '{} {}'.format(self.amount, self.instrument.symbol)

    def __repr__(self):
        return str(self)
