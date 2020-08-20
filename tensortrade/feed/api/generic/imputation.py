"""
imputation.py contains classes for imputation stream operations.
"""

import numpy as np

from tensortrade.feed.core.base import Stream, T


class ForwardFill(Stream[T]):
    """A stream operator that computes the forward fill imputation of a stream."""

    generic_name = "ffill"

    def __init__(self) -> None:
        super().__init__()
        self.previous = None

    def forward(self) -> T:
        node = self.inputs[0]
        if not self.previous or np.isfinite(node.value):
            self.previous = node.value
        return self.previous

    def has_next(self) -> bool:
        return True


class FillNa(Stream[T]):
    """A stream operator that computes the padded imputation of a stream.

    Parameters
    ----------
    fill_value : `T`
        The fill value to use for missing values in the stream.
    """

    generic_name = "fillna"

    def __init__(self, fill_value: T):
        super().__init__()
        self.fill_value = fill_value

    def forward(self) -> T:
        node = self.inputs[0]
        if np.isnan(node.value):
            return self.fill_value
        return node.value

    def has_next(self) -> bool:
        return True
