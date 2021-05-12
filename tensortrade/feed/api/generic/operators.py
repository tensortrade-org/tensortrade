"""
operators.py contains function for generic stream operators.
"""

from typing import Callable

from tensortrade.feed.core.base import Stream, T
from tensortrade.feed.core.operators import (
    Apply,
    Lag,
    Freeze,
    Accumulator,
    Copy,
    TypeVar
)

K = TypeVar("K")


@Stream.register_generic_method(["apply"])
def apply(s: "Stream[T]",
          func: Callable[[T], K],
          dtype: str = None) -> "Stream[K]":
    """Creates an apply stream.

    Parameters
    ----------
    s : `Stream[T]`
        A generic stream.
    func : `Callable[[T], K]`
        A function to be applied to the values of a stream.
    dtype : str, optional
        The data type of the values after function is applied.

    Returns
    -------
    `Stream[K]`
        A transformed stream of `s`.
    """
    if dtype is None:
        dtype = s.dtype
    return Apply(func, dtype=dtype)(s)


@Stream.register_generic_method(["lag"])
def lag(s: "Stream[T]", lag: int = 1, dtype: str = None) -> "Stream[T]":
    """Creates a lag stream.

    Parameters
    ----------
    s : `Stream[T]`
        A generic stream.
    lag : int, default 1
        The number of steps to lag behind by
    dtype : str, optional
        The data type of the stream

    Returns
    -------
    `Stream[T]`
        A lag stream of `s`.
    """
    if dtype is None:
        dtype = s.dtype
    return Lag(lag, dtype=dtype)(s)


@Stream.register_generic_method(["copy"])
def copy(s: "Stream[T]") -> "Stream[T]":
    """Creates a copy stream.

    Parameters
    ----------
    s : `Stream[T]`
        A generic stream.

    Returns
    -------
    `Stream[T]`
        A copy stream of `s`.
    """
    return Copy()(s).astype(s.dtype)


@Stream.register_generic_method(["freeze"])
def freeze(s: "Stream[T]") -> "Stream[T]":
    """Creates a frozen stream.

    Parameters
    ----------
    s : `Stream[T]`
        A generic stream.

    Returns
    -------
    `Stream[T]`
        A frozen stream of `s`.
    """
    return Freeze()(s).astype(s.dtype)


@Stream.register_generic_method(["accumulate"])
def accumulate(s: "Stream[T]", func: Callable[[T, T], T]) -> "Stream[T]":
    """Creates an accumulation stream.

    Parameters
    ----------
    s : `Stream[T]`
        A generic stream.
    func : `Callable[[T, T], T]`
        An accumulator function.

    Returns
    -------
    `Stream[T]`
        A accumulated stream of `s`.
    """
    return Accumulator(func)(s).astype(s.dtype)
