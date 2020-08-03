
import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.core.operators import BinOp
from tensortrade.feed.api.float import Float


@Float.register(["ceil"])
def ceil(s: "Stream[float]") -> "Stream[float]":
    """Computes the ceiling of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The ceiling stream of `s`.
    """
    return s.apply(np.ceil).astype("float")


@Float.register(["floor"])
def floor(s: "Stream[float]") -> "Stream[float]":
    """Computes the floor of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The floor stream of `s`.
    """
    return s.apply(np.floor).astype("float")


@Float.register(["sqrt"])
def sqrt(s: "Stream[float]") -> "Stream[float]":
    """Computes the square root of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The square root stream of `s`.
    """
    return s.apply(np.sqrt).astype("float")


@Float.register(["square"])
def square(s: "Stream[float]") -> "Stream[float]":
    """Computes the square of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The square stream of `s`.
    """
    return s.apply(np.square).astype("float")


@Float.register(["log"])
def log(s: "Stream[float]") -> "Stream[float]":
    """Computes the log of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The log stream of `s`.
    """
    return s.apply(np.log).astype("float")


@Float.register(["pct_change"])
def pct_change(s: "Stream[float]",
               periods: int = 1,
               fill_method: str = "pad") -> "Stream[float]":
    """Computes the percent change of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    periods : int, default 1
        The number of periods to lag for until computing the percent change.
    fill_method : str, default "pad"
        The fill method to use for missing values.

    Returns
    -------
    `Stream[float]`
        The percent change stream of `s`.
    """
    if fill_method is not None:
        assert fill_method in ["pad", "ffill"]
    if fill_method == "pad" or fill_method == "ffill":
        stream = s.ffill()
    else:
        stream = s
    change = (stream / stream.lag(periods)) - 1
    return change.astype("float")


@Float.register(["diff"])
def diff(s: "Stream[float]", periods: int = 1) -> "Stream[float]":
    """Computes the difference of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    periods : int, default 1
        The number of periods to lag for until computing the difference.

    Returns
    -------
    `Stream[float]`
        The difference stream of `s`.
    """
    return BinOp(np.subtract)(s, s.lag(periods)).astype("float")
