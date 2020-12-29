"""
expanding.py contains functions and classes for expanding stream operations.
"""

from typing import Callable, List

import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.float import Float


class ExpandingNode(Stream[float]):
    """A stream operator for aggregating an entire history of a stream.

    Parameters
    ----------
    func : `Callable[[List[float]], float]`
        A function that aggregates the history of a stream.
    """

    def __init__(self, func: "Callable[[List[float]], float]") -> None:
        super().__init__()
        self.func = func

    def forward(self) -> float:
        expanding = self.inputs[0]
        history = expanding.value
        if len(history) < expanding.min_periods:
            return np.nan
        return self.func(history)

    def has_next(self):
        return True


class ExpandingCount(ExpandingNode):
    """A stream operator that counts the number of non-missing values."""

    def __init__(self) -> None:
        super().__init__(lambda w: (~np.isnan(w)).sum())

    def forward(self) -> float:
        return self.func(self.inputs[0].value)


class Expanding(Stream[List[float]]):
    """A stream that generates the entire history of a stream at each time step.

    Parameters
    ----------
    min_periods : int, default 1
        The number of periods to wait before producing values from the aggregation
        function.
    """

    generic_name = "expanding"

    def __init__(self, min_periods: int = 1) -> None:
        super().__init__()
        self.min_periods = min_periods
        self.history = []

    def forward(self) -> "List[float]":
        v = self.inputs[0].value
        if not np.isnan(v):
            self.history += [v]
        return self.history

    def has_next(self) -> bool:
        return True

    def agg(self, func: Callable[[List[float]], float]) -> "Stream[float]":
        """Computes an aggregation of a stream's history.

        Parameters
        ----------
        func : `Callable[[List[float]], float]`
            A aggregation function.

        Returns
        -------
        `Stream[float]`
            A stream producing aggregations of the stream history at each time
            step.
        """
        return ExpandingNode(func)(self).astype("float")

    def count(self) -> "Stream[float]":
        """Computes an expanding count fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding count stream.
        """
        return ExpandingCount()(self).astype("float")

    def sum(self) -> "Stream[float]":
        """Computes an expanding sum fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding sum stream.
        """
        return self.agg(np.sum).astype("float")

    def mean(self) -> "Stream[float]":
        """Computes an expanding mean fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding mean stream.
        """
        return self.agg(np.mean).astype("float")

    def var(self) -> "Stream[float]":
        """Computes an expanding variance fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding variance stream.
        """
        return self.agg(lambda x: np.var(x, ddof=1)).astype("float")

    def median(self) -> "Stream[float]":
        """Computes an expanding median fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding median stream.
        """
        return self.agg(np.median).astype("float")

    def std(self) -> "Stream[float]":
        """Computes an expanding standard deviation fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding standard deviation stream.
        """
        return self.agg(lambda x: np.std(x, ddof=1)).astype("float")

    def min(self) -> "Stream[float]":
        """Computes an expanding minimum fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding minimum stream.
        """
        return self.agg(np.min).astype("float")

    def max(self) -> "Stream[float]":
        """Computes an expanding maximum fo the underlying stream.

        Returns
        -------
        `Stream[float]`
            An expanding maximum stream.
        """
        return self.agg(np.max).astype("float")

    def reset(self) -> None:
        self.history = []
        super().reset()


@Float.register(["expanding"])
def expanding(s: "Stream[float]", min_periods: int = 1) -> "Stream[List[float]]":
    """Computes a stream that generates the entire history of a stream at each
    time step.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    min_periods : int, default 1
        The number of periods to wait before producing values from the aggregation
        function.
    """
    return Expanding(
        min_periods=min_periods
    )(s)
