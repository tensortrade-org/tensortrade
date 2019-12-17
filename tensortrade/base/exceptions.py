

# =============================================================================
# Quantity Exceptions
# =============================================================================
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


class QuantityOpPathMismatch(Exception):

    def __init(self, left_id, right_id, *args):
        super().__init__(
            "Invalid operation between quantities with unequal path id: {} {}.".format(left_id, right_id),
            *args
        )


# =============================================================================
# Instrument Exceptions
# =============================================================================
class IncompatibleInstrumentOperation(Exception):

    def __init__(self, left, right, *args):
        super().__init__(
            "Instruments are not of the same type ({} and {}).".format(left, right),
            *args
        )


# =============================================================================
# Order Exceptions
# =============================================================================
class InvalidOrderQuantity(Exception):

    def __init__(self, size, *args):
        super().__init__(
            "Invalid Quantity: {}. Order sizes must be positive.".format(size),
            *args
        )


class IncompatibleRecipePath(Exception):

    def __init__(self, order, recipe, *args):
        super().__init__(
            "Incompatible {} following {}.".format(order, recipe),
            *args
        )


# =============================================================================
# Wallet Exceptions
# =============================================================================
class InsufficientFundsForAllocation(Exception):

    def __init__(self, balance, size, *args):
        super().__init__(
            "Insufficient funds for allocating size {} with balance {}.".format(size, balance),
            *args
        )


# =============================================================================
# Trading Pair Exceptions
# =============================================================================
class InvalidTradingPair(Exception):

    def __init__(self, base, quote,*args):
        super().__init__(
            "Invalid instrument pair {}/{}.".format(base, quote),
            *args
        )
