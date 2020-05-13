
import numpy as np

from tensortrade.data.feed.core import Stream
from tensortrade.data.feed.api.float import Float


class CumSum(Stream[float]):

    def __init__(self):
        super().__init__()
        self.c_sum = 0

    def forward(self):
        node = self.inputs[0]
        if np.isnan(node.value):
            return np.nan
        self.c_sum += node.value
        return self.c_sum

    def has_next(self):
        return True


class CumProd(Stream[float]):

    def __init__(self):
        super().__init__()
        self.c_prod = 1

    def forward(self):
        node = self.inputs[0]
        if np.isnan(node.value):
            return np.nan
        self.c_prod *= node.value
        return self.c_prod

    def has_next(self):
        return True


class CumMin(Stream[float]):

    def __init__(self, skip_na=True):
        super().__init__()
        self.skip_na = skip_na
        self.c_min = np.inf

    def forward(self):
        node = self.inputs[0]
        if self.skip_na:
            if np.isnan(node.value):
                return np.nan
            if not np.isnan(node.value) and node.value < self.c_min:
                self.c_min = node.value
        else:
            if self.c_min is None:
                self.c_min = node.value
            elif np.isnan(node.value):
                self.c_min = np.nan
            elif node.value < self.c_min:
                self.c_min = node.value
        return self.c_min

    def has_next(self):
        return True


class CumMax(Stream[float]):

    def __init__(self, skip_na: bool = True):
        super().__init__()
        self.skip_na = skip_na
        self.c_max = -np.inf

    def forward(self):
        node = self.inputs[0]
        if self.skip_na:
            if np.isnan(node.value):
                return np.nan
            if not np.isnan(node.value) and node.value > self.c_max:
                self.c_max = node.value
        else:
            if self.c_max is None:
                self.c_max = node.value
            elif np.isnan(node.value):
                self.c_max = np.nan
            elif node.value > self.c_max:
                self.c_max = node.value
        return self.c_max

    def has_next(self):
        return True


@Float.register(["cumsum"])
def cumsum(s: "Stream[float]") -> "Stream[float]":
    return CumSum()(s).astype("float")


@Float.register(["cumprod"])
def cumprod(s: "Stream[float]") -> "Stream[float]":
    return CumProd()(s).astype("float")


@Float.register(["cummin"])
def cummin(s: "Stream[float]", skipna: bool = True) -> "Stream[float]":
    return CumMin(skip_na=skipna)(s).astype("float")


@Float.register(["cummax"])
def cummax(s: "Stream[float]", skipna: bool = True) -> "Stream[float]":
    return CumMax(skip_na=skipna)(s).astype("float")
