
import numpy as np

from tensortrade.feed import Stream, T


class WarmUp(Stream[T]):

    def __init__(self, periods: int):
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
    return WarmUp(periods=periods)(s)
