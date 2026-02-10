"""Holds all the exceptions for the project."""

from numbers import Number


# =============================================================================
# Quantity Exceptions
# =============================================================================
class InvalidNegativeQuantity(Exception):
    """Raised when a `Quantity` tries to be instantiated with a negative amount.

    Parameters
    ----------
    size : float
        The size that was specified for the `Quantity`.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, size: float, *args) -> None:
        super().__init__(
            f"Invalid Quantity: {size}. Amounts cannot be negative.", *args
        )


class InvalidNonNumericQuantity(Exception):
    """Raised when a `Quantity` tries to be instantiated with a value
    that is not numeric.

    Parameters
    ----------
    size : `Union[float, int, Number]`
        The value that was specified for the `Quantity`.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, size: float | int | Number, *args) -> None:
        super().__init__(
            f"Invalid Quantity: {size}. Amounts cannot be non-numeric.", *args
        )


class QuantityOpPathMismatch(Exception):
    """Raised when an operation tries to occur between quantities that are not
    under the same path_id.

    Parameters
    ----------
    left_id : str
        The path_id for the left argument in the operation.
    right_id : str
        The path_id for the right argument in the operation.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, left_id: str, right_id: str, *args) -> None:
        super().__init__(
            f"Invalid operation between quantities with unequal path id: {left_id} {right_id}.",
            *args,
        )


class DoubleLockedQuantity(Exception):
    """Raised when a locked `Quantity` is trying to get locked again.

    Parameters
    ----------
    quantity : `Quantity`
        A locked quantity.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, quantity: "Quantity", *args) -> None:
        super().__init__(
            f"Cannot lock quantity that has previously been locked: {quantity}.", *args
        )


class DoubleUnlockedQuantity(Exception):
    """Raised when a free `Quantity` is trying to get unlocked.

    Parameters
    ----------
    quantity : `Quantity`
        A unlocked quantity.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, quantity: "Quantity", *args) -> None:
        super().__init__(
            f"Cannot unlock quantity that has previously been unlocked: {quantity}.",
            *args,
        )


class QuantityNotLocked(Exception):
    """Raised when a locked `Quantity` does not have a path_id in the `Wallet`
    it is trying to be unlocked in.

    Parameters
    ----------
    quantity : `Quantity`
        A locked quantity.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, quantity: "Quantity", *args) -> None:
        super().__init__(
            f"Cannot unlock quantity that has not been locked in this wallet: {quantity}.",
            *args,
        )


# =============================================================================
# Instrument Exceptions
# =============================================================================
class IncompatibleInstrumentOperation(Exception):
    """Raised when two quantities with different instruments occurs.

    Parameters
    ----------
    left : `Quantity`
        The left argument of the operation.
    right : `Quantity`
        The right argument of the operation.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, left: "Quantity", right: "Quantity", *args) -> None:
        super().__init__(
            f"Instruments are not of the same type ({left} and {right}).", *args
        )


# =============================================================================
# Order Exceptions
# =============================================================================
class InvalidOrderQuantity(Exception):
    """Raised when an `Order` with a non-negative amount is placed

    Parameters
    ----------
    quantity : `Quantity`
        An invalid order quantity.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, quantity: "Quantity", *args) -> None:
        super().__init__(
            f"Invalid Quantity: {quantity}. Order sizes must be positive.", *args
        )


# =============================================================================
# Wallet Exceptions
# =============================================================================
class InsufficientFunds(Exception):
    """Raised when requested funds are greater than the free balance of a `Wallet`

    Parameters
    ----------
    balance : `Quantity`
        The balance of the `Wallet` where funds are being allocated from.
    size : `Quantity`
        The amount being requested for allocation.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, balance: "Quantity", size: "Quantity", *args) -> None:
        super().__init__(
            f"Insufficient funds for allocating size {size} with balance {balance}.",
            *args,
        )


# =============================================================================
# Trading Pair Exceptions
# =============================================================================
class InvalidTradingPair(Exception):
    """Raised when an invalid trading pair is trying to be created.

    Parameters
    ----------
    base : 'Instrument'
        The base instrument of the pair.
    quote : 'Instrument'
        The quote instrument of the pair.
    *args : positional arguments
        More positional arguments for the exception.
    """

    def __init__(self, base: "Instrument", quote: "Instrument", *args) -> None:
        super().__init__(f"Invalid instrument pair {base}/{quote}.", *args)
