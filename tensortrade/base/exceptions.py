

class InvalidNegativeQuantity(Exception):

    def __init__(self, size, *args):
        super().__init__(
            "Invalid Quantity: {}. Amounts cannot be negative.".format(size),
            *args
        )


class InvalidNonNumericQuantity(Exception):

    def __init(self, size, *args):
        super().__init__(
            "Invalid Quantity: {}. Amounts cannot be non-numeric.".format(size),
            *args
        )


class IncompatibleInstrumentOperation(Exception):

    def __init__(self, left, right, *args):
        super().__init__(
            "Instruments are not of the same type ({} and {}).".format(left, right),
            *args
        )


class InvalidOrderQuantity(Exception):

    def __init__(self, size, *args):
        super().__init__(
            "Invalid Quantity: {}. Order sizes must be positive.".format(size),
            *args
        )


class InsufficientFunds(Exception):

    def __init__(self, balance, size, *args):
        super().__init__(
            "Insufficient funds for size {} with balance {}.".format(size, balance),
            *args
        )


class IncompatibleRecipePath:

    def __init__(self, order, recipe, *args):
        super().__init__(
            "Incompatible {} following {}.".format(order, recipe)
        )
