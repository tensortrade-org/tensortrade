

from tensortrade.data.feed.core.base import Stream, T
from tensortrade.data.feed.core.operators import (
    Apply,
    Lag,
    Select,
    Freeze,
    Accumulator,
    Copy
)

from typing import Callable, Iterable


@Stream.register_generic_method(["apply"])
def apply(s: "Stream[T]", func: Callable, dtype=None) -> "Stream[T]":
    if dtype is None:
        dtype = s.dtype
    return Apply(func, dtype=dtype)(s)


@Stream.register_generic_method(["lag"])
def lag(s: "Stream[T]", lag: int = 1, dtype=None) -> "Stream[T]":
    if dtype is None:
        dtype = s.dtype
    return Lag(lag, dtype=dtype)(s)


@Stream.register_generic_method(["copy"])
def copy(s: "Stream[T]") -> "Stream[T]":
    return Copy()(s).astype(s.dtype)


@Stream.register_generic_method(["freeze"])
def freeze(s: "Stream[T]") -> "Stream[T]":
    return Freeze()(s).astype(s.dtype)


@Stream.register_generic_method(["accumulate"])
def accumulate(s: "Stream[T]", func: Callable[[T, T], T]) -> "Stream[T]":
    return Accumulator(func)(s).astype(s.dtype)
