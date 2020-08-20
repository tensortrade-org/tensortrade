"""
warmup.py contains classes for warm up stream operations.
"""

import numpy as np

from tensortrade.feed.core.base import Stream, T


class WarmUp(Stream[T]):
    """A stream operator for warming up a given stream.

    Parameters
    ----------
    periods : int
        Number of periods to warm up.
    """

    def __init__(self, periods: int) -> None:
        super().__init__()
        self.count = 0
        self.periods = periods

    def forward(self) -> T:
        v = self.inputs[0].value
        if self.count < self.periods:
            self.count += 1
            return np.nan
        return v

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        self.count = 0


@Stream.register_generic_method(["warmup"])
def warmup(s: "Stream[T]", periods: int) -> "Stream[T]":
    """Creates a warmup stream.

    Parameters
    ----------
    s : `Stream[T]`
        A generic stream.
    periods : int
        Number of periods to warm up.

    Returns
    -------
    `Stream[T]`
        The warmup stream of `s`.
    """
    return WarmUp(periods=periods)(s)
