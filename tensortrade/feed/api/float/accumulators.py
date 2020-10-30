
import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.float import Float


class CumSum(Stream[float]):
    """A stream operator that creates a cumulative sum of values.

    References
    ----------
    .. [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cumsum.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.c_sum = 0

    def forward(self) -> float:
        node = self.inputs[0]
        if np.isnan(node.value):
            return np.nan
        self.c_sum += node.value
        return self.c_sum

    def has_next(self) -> bool:
        return True


class CumProd(Stream[float]):
    """A stream operator that creates a cumulative product of values.

    References
    ----------
    .. [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cumprod.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.c_prod = 1

    def forward(self) -> float:
        node = self.inputs[0]
        if np.isnan(node.value):
            return np.nan
        self.c_prod *= node.value
        return self.c_prod

    def has_next(self) -> bool:
        return True


class CumMin(Stream[float]):
    """A stream operator that creates a cumulative minimum of values.

    Parameters
    ----------
    skip_na : bool, default True
        Exclude NA/null values. If a value is NA, the result will be NA.

    References
    ----------
    [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cummin.html
    """

    def __init__(self, skip_na: bool = True) -> None:
        super().__init__()
        self.skip_na = skip_na
        self.c_min = np.inf

    def forward(self) -> float:
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

    def has_next(self) -> bool:
        return True


class CumMax(Stream[float]):
    """A stream operator that creates a cumulative maximum of values.

    Parameters
    ----------
    skip_na : bool, default True
        Exclude NA/null values. If a value is NA, the result will be NA.

    References
    ----------
    [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cummax.html
    """

    def __init__(self, skip_na: bool = True) -> None:
        super().__init__()
        self.skip_na = skip_na
        self.c_max = -np.inf

    def forward(self) -> float:
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

    def has_next(self) -> bool:
        return True


@Float.register(["cumsum"])
def cumsum(s: "Stream[float]") -> "Stream[float]":
    """Computes the cumulative sum of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The cumulative sum stream of `s`.
    """
    return CumSum()(s).astype("float")


@Float.register(["cumprod"])
def cumprod(s: "Stream[float]") -> "Stream[float]":
    """Computes the cumulative product of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The cumulative product stream of `s`.
    """
    return CumProd()(s).astype("float")


@Float.register(["cummin"])
def cummin(s: "Stream[float]", skipna: bool = True) -> "Stream[float]":
    """Computes the cumulative minimum of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    skipna : bool, default True
        Exclude NA/null values. If a value is NA, the result will be NA.

    Returns
    -------
    `Stream[float]`
        The cumulative minimum stream of `s`.
    """
    return CumMin(skip_na=skipna)(s).astype("float")


@Float.register(["cummax"])
def cummax(s: "Stream[float]", skipna: bool = True) -> "Stream[float]":
    """Computes the cumulative maximum of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    skipna : bool, default True
        Exclude NA/null values. If a value is NA, the result will be NA.

    Returns
    -------
    `Stream[float]`
        The cumulative maximum stream of `s`.
    """
    return CumMax(skip_na=skipna)(s).astype("float")
