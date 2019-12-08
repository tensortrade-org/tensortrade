

class NegativeQuantityException(Exception):

    def __init__(self, amount, *args):
        super().__init__(
            "Invalid Quantity: {}. Amounts cannot be negative.".format(amount),
            *args
        )


class IncompatibleInstrumentOperation(Exception):

    def __init__(self, left, right, *args):
        super().__init__(
            "Instruments are not of the same type ({} and {}).".format(left, right),
            *args
        )


class InsufficientFunds(Exception):

    def __init__(self, balance, amount, *args):
        super().__init__(
            "Insufficient funds for amount {} with balance {}.".format(amount, balance),
            *args
        )
