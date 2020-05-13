

import numpy as np

from typing import Callable, List

from tensortrade.data.feed.core import Stream
from tensortrade.data.feed.api.float import Float


class ExpandingNode(Stream[float]):

    def __init__(self, func):
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

    def __init__(self):
        super().__init__(lambda w: (~np.isnan(w)).sum())

    def forward(self) -> float:
        return self.func(self.inputs[0].value)


class Expanding(Stream[List[float]]):

    generic_name = "expanding"

    def __init__(self, min_periods: int = 1):
        super().__init__()
        self.min_periods = min_periods
        self.history = []

    def forward(self):
        v = self.inputs[0].value
        if not np.isnan(v):
            self.history += [v]
        return self.history

    def has_next(self):
        return True

    def agg(self, func: Callable[[List[float]], float]) -> "Stream[float]":
        return ExpandingNode(func)(self).astype("float")

    def count(self) -> "Stream[float]":
        return ExpandingCount()(self).astype("float")

    def sum(self) -> "Stream[float]":
        return self.agg(np.sum).astype("float")

    def mean(self) -> "Stream[float]":
        return self.agg(np.mean).astype("float")

    def var(self) -> "Stream[float]":
        return self.agg(lambda x: np.var(x, ddof=1)).astype("float")

    def median(self) -> "Stream[float]":
        return self.agg(np.median).astype("float")

    def std(self) -> "Stream[float]":
        return self.agg(lambda x: np.std(x, ddof=1)).astype("float")

    def min(self) -> "Stream[float]":
        return self.agg(np.min).astype("float")

    def max(self) -> "Stream[float]":
        return self.agg(np.max).astype("float")


@Float.register(["expanding"])
def expanding(s: "Stream[float]", min_periods: int = 1) -> "Stream[List[float]]":
    return Expanding(
        min_periods=min_periods
    )(s)
