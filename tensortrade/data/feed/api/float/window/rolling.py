
import numpy as np

from typing import List

from tensortrade.data.feed.core import Stream
from tensortrade.data.feed.api.float import Float


class RollingNode(Stream[float]):

    def __init__(self, func):
        super().__init__(dtype="float")
        self.func = func
        self.n = 0

    def forward(self):
        rolling = self.inputs[0]
        history = rolling.value

        output = np.nan if rolling.n - rolling.nan < rolling.min_periods else self.func(history)

        return output

    def has_next(self):
        return True


class RollingCount(RollingNode):

    def __init__(self):
        super().__init__(lambda w: (~np.isnan(w)).sum())

    def forward(self):
        rolling = self.inputs[0]
        history = rolling.value

        if len(history) < rolling.min_periods:
            return np.nan
        return self.func(history)


class Rolling(Stream[List[float]]):

    generic_name = "rolling"

    def __init__(self, window: int, min_periods: int = 1):
        super().__init__()
        assert min_periods <= window
        self.window = window
        self.min_periods = min_periods

        self.n = 0
        self.nan = 0

        self.history = []

    def forward(self):
        node = self.inputs[0]

        self.n += 1
        self.nan += int(node.value != node.value)

        self.history.insert(0, node.value)

        if len(self.history) > self.window:
            self.history.pop()
        return self.history

    def has_next(self):
        return True

    def agg(self, func) -> "Stream[float]":
        return RollingNode(func)(self).astype("float")

    def count(self) -> "Stream[float]":
        return RollingCount()(self).astype("float")

    def sum(self) -> "Stream[float]":
        func = np.nansum if self.min_periods < self.window else np.sum
        return self.agg(func).astype("float")

    def mean(self) -> "Stream[float]":
        func = np.nanmean if self.min_periods < self.window else np.mean
        return self.agg(func).astype("float")

    def var(self) -> "Stream[float]":
        func1 = lambda x: np.nanvar(x, ddof=1)
        func2 = lambda x: np.var(x, ddof=1)
        func = func1 if self.min_periods < self.window else func2
        return self.agg(func).astype("float")

    def median(self) -> "Stream[float]":
        func = np.nanmedian if self.min_periods < self.window else np.median
        return self.agg(func).astype("float")

    def std(self) -> "Stream[float]":
        return self.var().sqrt()

    def min(self) -> "Stream[float]":
        func = np.nanmin if self.min_periods < self.window else np.min
        return self.agg(func).astype("float")

    def max(self) -> "Stream[float]":
        func = np.nanmax if self.min_periods < self.window else np.max
        return self.agg(func).astype("float")


@Float.register(["rolling"])
def rolling(s: "Stream[float]", window: int, min_periods: int = 1) -> "Stream[List[float]]":
    return Rolling(
        window=window,
        min_periods=min_periods
    )(s)
