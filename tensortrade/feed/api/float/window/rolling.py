"""
rolling.py contains functions and classes for rolling stream operations.
"""

from typing import List, Callable

import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.float import Float


class RollingNode(Stream[float]):
    """A stream operator for aggregating a rolling window of a stream.

    Parameters
    ----------
    func : `Callable[[List[float]], float]`
        A function that aggregates a rolling window.
    """

    def __init__(self, func: "Callable[[List[float]], float]"):
        super().__init__(dtype="float")
        self.func = func
        self.n = 0

    def forward(self) -> float:
        rolling = self.inputs[0]
        history = rolling.value

        output = np.nan if rolling.n - rolling.nan < rolling.min_periods else self.func(history)

        return output

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        self.n = 0
        super().reset()


class RollingCount(RollingNode):
    """A stream operator that counts the number of non-missing values in the
    rolling window."""

    def __init__(self):
        super().__init__(lambda w: (~np.isnan(w)).sum())

    def forward(self):
        rolling = self.inputs[0]
        history = rolling.value
        return self.func(history)


class Rolling(Stream[List[float]]):
    """A stream that generates a rolling window of values from a stream.

    Parameters
    ----------
    window : int
        The size of the rolling window.
    min_periods : int, default 1
        The number of periods to wait before producing values from the aggregation
        function.
    """

    generic_name = "rolling"

    def __init__(self,
                 window: int,
                 min_periods: int = 1) -> None:
        super().__init__()
        assert min_periods <= window
        self.window = window
        self.min_periods = min_periods

        self.n = 0
        self.nan = 0

        self.history = []

    def forward(self) -> "List[float]":
        node = self.inputs[0]

        self.n += 1
        self.nan += int(node.value != node.value)

        self.history.insert(0, node.value)

        if len(self.history) > self.window:
            self.history.pop()
        return self.history

    def has_next(self) -> bool:
        return True

    def agg(self, func: "Callable[[List[float]], float]") -> "Stream[float]":
        """Computes an aggregation of a rolling window of values.

        Parameters
        ----------
        func : `Callable[[List[float]], float]`
            A aggregation function.

        Returns
        -------
        `Stream[float]`
            A stream producing aggregations of a rolling window of values.
        """
        return RollingNode(func)(self).astype("float")

    def count(self) -> "Stream[float]":
        """Computes a rolling count from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling count stream.
        """
        return RollingCount()(self).astype("float")

    def sum(self) -> "Stream[float]":
        """Computes a rolling sum from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling sum stream.
        """
        func = np.nansum if self.min_periods < self.window else np.sum
        return self.agg(func).astype("float")

    def mean(self) -> "Stream[float]":
        """Computes a rolling mean from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling mean stream.
        """
        func = np.nanmean if self.min_periods < self.window else np.mean
        return self.agg(func).astype("float")

    def var(self) -> "Stream[float]":
        """Computes a rolling variance from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling variance stream.
        """
        def func1(x): return np.nanvar(x, ddof=1)
        def func2(x): return np.var(x, ddof=1)
        func = func1 if self.min_periods < self.window else func2
        return self.agg(func).astype("float")

    def median(self) -> "Stream[float]":
        """Computes a rolling median from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling median stream.
        """
        func = np.nanmedian if self.min_periods < self.window else np.median
        return self.agg(func).astype("float")

    def std(self) -> "Stream[float]":
        """Computes a rolling standard deviation from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling standard deviation stream.
        """
        return self.var().sqrt()

    def min(self) -> "Stream[float]":
        """Computes a rolling minimum from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling minimum stream.
        """
        func = np.nanmin if self.min_periods < self.window else np.min
        return self.agg(func).astype("float")

    def max(self) -> "Stream[float]":
        """Computes a rolling maximum from the underlying stream.

        Returns
        -------
        `Stream[float]`
            A rolling maximum stream.
        """
        func = np.nanmax if self.min_periods < self.window else np.max
        return self.agg(func).astype("float")

    def reset(self) -> None:
        self.n = 0
        self.nan = 0
        self.history = []
        super().reset()


@Float.register(["rolling"])
def rolling(s: "Stream[float]",
            window: int,
            min_periods: int = 1) -> "Stream[List[float]]":
    """Creates a stream that generates a rolling window of values from a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    window : int
        The size of the rolling window.
    min_periods : int, default 1
        The number of periods to wait before producing values from the aggregation
        function.
    """
    return Rolling(
        window=window,
        min_periods=min_periods
    )(s)
