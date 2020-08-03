
import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.core.operators import BinOp
from tensortrade.feed.api.float import Float


@Float.register(["clamp_min"])
def clamp_min(s: "Stream[float]", c_min: float) -> "Stream[float]":
    """Clamps the minimum value of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    c_min : float
        The mimimum value to clamp by.

    Returns
    -------
    `Stream[float]`
        The minimum clamped stream of `s`.
    """
    return BinOp(np.maximum)(s, Stream.constant(c_min)).astype("float")


@Float.register(["clamp_max"])
def clamp_max(s: "Stream[float]", c_max: float) -> "Stream[float]":
    """Clamps the maximum value of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    c_max : float
        The maximum value to clamp by.

    Returns
    -------
    `Stream[float]`
        The maximum clamped stream of `s`.
    """
    return BinOp(np.minimum)(s, Stream.constant(c_max)).astype("float")


@Float.register(["clamp"])
def clamp(s: "Stream[float]", c_min: float, c_max: float) -> "Stream[float]":
    """Clamps the minimum and maximum value of a stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    c_min : float
        The mimimum value to clamp by.
    c_max : float
        The maximum value to clamp by.

    Returns
    -------
    `Stream[float]`
        The clamped stream of `s`.
    """
    stream = s.clamp_min(c_min).astype("float")
    stream = stream.clamp_max(c_max).astype("float")
    return stream


@Float.register(["min"])
def min(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the minimum of two streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]`
        The second float stream.

    Returns
    -------
    `Stream[float]`
        The minimum stream of `s1` and `s2`.
    """
    return BinOp(np.minimum)(s1, s2).astype("float")


@Float.register(["max"])
def max(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the maximum of two streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]`
        The second float stream.

    Returns
    -------
    `Stream[float]`
        The maximum stream of `s1` and `s2`.
    """
    return BinOp(np.maximum)(s1, s2).astype("float")
