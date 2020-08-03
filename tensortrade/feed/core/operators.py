
from typing import Callable, TypeVar


import numpy as np

from tensortrade.feed.core.base import Stream, T


K = TypeVar("K")


class Apply(Stream[K]):
    """An operator stream that applies a specific function to the values of
    a given stream.

    Parameters
    ----------
    func : `Callable[[T], ...]`
        A function to be applied to the values of a stream.
    dtype : str, optional
        The data type of the values after function is applied.
    """

    def __init__(self,
                 func: Callable[[T], K],
                 dtype: str = None) -> None:
        super().__init__(dtype=dtype)
        self.func = func

    def forward(self) -> K:
        node = self.inputs[0]
        return self.func(node.value)

    def has_next(self) -> bool:
        return True


class Lag(Stream[T]):
    """An operator stream that returns the lagged value of a given stream.

    Parameters
    ----------
    lag : int
        The number of steps to lag behind by
    dtype : str, optional
        The data type of the stream
    """

    generic_name = "lag"

    def __init__(self,
                 lag: int = 1,
                 dtype: str = None) -> None:
        super().__init__(dtype=dtype)
        self.lag = lag
        self.runs = 0
        self.history = []

    def forward(self) -> T:
        node = self.inputs[0]
        if self.runs < self.lag:
            self.runs += 1
            self.history.insert(0, node.value)
            return np.nan

        self.history.insert(0, node.value)
        return self.history.pop()

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        self.runs = 0
        self.history = []


class Accumulator(Stream[T]):
    """An operator stream that accumulates values of a given stream.

    Parameters
    ----------
    func : Callable[[T,T], T]
        An accumulator function.
    dtype : str
        The data type of accumulated value.
    """

    def __init__(self,
                 func: "Callable[[T, T], T]",
                 dtype: str = None) -> None:
        super().__init__(dtype)
        self.func = func
        self.past = None

    def forward(self):
        node = self.inputs[0]
        if self.past is None:
            self.past = node.value
            return self.past
        v = self.func(self.past, node.value)
        self.past = v
        return v

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        self.past = None


class Copy(Stream[T]):
    """A stream operator that copies the values of a given stream."""

    generic_name = "copy"

    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> T:
        return self.inputs[0].value

    def has_next(self) -> bool:
        return True


class Freeze(Stream[T]):
    """A stream operator that freezes the value of a given stream and generates
    that value."""

    generic_name = "freeze"

    def __init__(self) -> None:
        super().__init__()
        self.freeze_value = None

    def forward(self) -> T:
        node = self.inputs[0]
        if not self.freeze_value:
            self.freeze_value = node.value
        return self.freeze_value

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        self.freeze_value = None


class BinOp(Stream[T]):
    """A stream operator that combines the values of two given streams into
    one value of the same type.

    Parameters
    ----------
    op : `Callable[[T, T], T]`
        The binary operation to be applied.
    dtype : str, optional
        The data type of the stream.
    """

    generic_name = "bin_op"

    def __init__(self,
                 op: Callable[[T, T], T],
                 dtype: str = None) -> None:
        super().__init__(dtype=dtype)
        self.op = op

    def forward(self) -> T:
        return self.op(self.inputs[0].value, self.inputs[1].value)

    def has_next(self) -> bool:
        return True
