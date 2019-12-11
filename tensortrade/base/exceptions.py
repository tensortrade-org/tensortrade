

class InvalidNegativeQuantity(Exception):

    def __init__(self, amount, *args):
        super().__init__(
            "Invalid Quantity: {}. Amounts cannot be negative.".format(amount),
            *args
        )


class InvalidNonNumericQuantity(Exception):

    def __init(self, amount, *args):
        super().__init__(
            "Invalid Quantity: {}. Amounts cannot be non-numeric.".format(amount),
            *args
        )


class IncompatibleInstrumentOperation(Exception):

    def __init__(self, left, right, *args):
        super().__init__(
            "Instruments are not of the same type ({} and {}).".format(left, right),
            *args
        )


class InvalidOrderQuantity(Exception):

    def __init__(self, amount, *args):
        super().__init__(
            "Invalid Quantity: {}. Order amounts must be positive.".format(amount),
            *args
        )


class InsufficientFunds(Exception):

    def __init__(self, balance, amount, *args):
        super().__init__(
            "Insufficient funds for amount {} with balance {}.".format(amount, balance),
            *args
        )


class IncompatibleRecipePath:

    def __init__(self, order, recipe, *args):
        super().__init__(
            "Incompatible {} following {}.".format(order, recipe)
        )
