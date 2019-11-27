

class Quantity:

    def __init__(self, amount, instrument):
        self._amount = round(amount, instrument.precision)
        self._instrument = instrument
        self.order = None

    @property
    def amount(self):
        return self._amount

    @property
    def instrument(self):
        return self._instrument

    def lock_on(self, order):
        self.order = order

    def locked_by(self) -> 'Order':
        return self.order

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
            raise Exception("Can't mulitply with non-numeric quantity.")
        other = float(other)
        return Quantity(self.amount * other, self.instrument)

    def __str__(self):
        return '{} {}'.format(self.amount, self.instrument.symbol)

    def __repr__(self):
        return str(self)