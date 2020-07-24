
import numpy as np

from tensortrade.feed import Stream, T
from typing import List, Callable


class Aggregate(Stream[T]):

    generic_name = "reduce"

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self) -> T:
        return self.func([s.value for s in self.inputs])

    def has_next(self):
        return True


class Reduce(Stream[List[T]]):

    def __init__(self, dtype: str = None):
        super().__init__(dtype=dtype)

    def forward(self) -> T:
        return [s.value for s in self.inputs]

    def has_next(self) -> bool:
        return True

    def agg(self, func: "Callable[[List[T]], T]") -> "Stream[T]":
        return Aggregate(func)(*self.inputs).astype(self.inputs[0].dtype)

    def sum(self) -> "Stream[T]":
        return self.agg(np.sum)

    def min(self) -> "Stream[T]":
        return self.agg(np.min)

    def max(self) -> "Stream[T]":
        return self.agg(np.max)

    def prod(self) -> "Stream[T]":
        return self.agg(np.prod)


@Stream.register_generic_method(["reduce"])
def reduce(streams: "List[Stream[T]]") -> "Stream[List[T]]":
    return Reduce()(*streams)
