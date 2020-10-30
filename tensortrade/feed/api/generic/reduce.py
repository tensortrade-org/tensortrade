"""
reduce.py contains functions and classes for reducing multiple streams
into a single stream.
"""

from typing import List, Callable

import numpy as np

from tensortrade.feed.core.base import Stream, T


class Aggregate(Stream[T]):
    """A multi-stream operator for aggregating multiple streams into a single stream.

    Parameters
    ----------
    func : `Callable[[List[Stream]], T]`
        A function for aggregating the value of multiple streams.
    """

    generic_name = "reduce"

    def __init__(self, func: Callable[[List[T]], T]):
        super().__init__()
        self.func = func

    def forward(self) -> T:
        return self.func([s.value for s in self.inputs])

    def has_next(self) -> bool:
        return True


class Reduce(Stream[list]):
    """A stream for reducing multiple streams of the same type.

    Parameters
    ----------
    dtype : str, optional
        The data type of the aggregated stream.
    """

    def __init__(self, dtype: str = None):
        super().__init__(dtype=dtype)

    def forward(self) -> "List[T]":
        return [s.value for s in self.inputs]

    def has_next(self) -> bool:
        return True

    def agg(self, func: "Callable[[List[T]], T]") -> "Stream[T]":
        """Computes the aggregation of the input streams.

        Returns
        -------
        `Stream[T]`
            An aggregated stream of the input streams.
        """
        return Aggregate(func)(*self.inputs).astype(self.inputs[0].dtype)

    def sum(self) -> "Stream[T]":
        """Computes the reduced sum of the input streams.

        Returns
        -------
        `Stream[T]`
            A reduced sum stream.
        """
        return self.agg(np.sum)

    def min(self) -> "Stream[T]":
        """Computes the reduced minimum of the input streams.

        Returns
        -------
        `Stream[T]`
            A reduced minimum stream.
        """
        return self.agg(np.min)

    def max(self) -> "Stream[T]":
        """Computes the reduced maximum of the input streams.

        Returns
        -------
        `Stream[T]`
            A reduced maximum stream.
        """
        return self.agg(np.max)

    def prod(self) -> "Stream[T]":
        """Computes the reduced product of the input streams.

        Returns
        -------
        `Stream[T]`
            A reduced product stream.
        """
        return self.agg(np.prod)


@Stream.register_generic_method(["reduce"])
def reduce(streams: "List[Stream[T]]") -> "Stream[List[T]]":
    """Creates a reduce stream from given input streams.

    Parameters
    ----------
    streams : `List[Stream[T]]`
        A list of input streams to be aggregated.

    Returns
    -------
    `Stream[List[T]]
        A reduce stream that generates a list of values all from the input
        streams.
    """
    return Reduce()(*streams)
