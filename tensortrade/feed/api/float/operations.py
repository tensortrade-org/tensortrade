"""
operations.py contains functions for computing arithmetic operations on float
streams.
"""

import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.core.operators import BinOp
from tensortrade.feed.api.float import Float


@Float.register(["add", "__add__"])
def add(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the addition of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from adding `s1` and `s2`.
    """
    if np.isscalar(s2):
        s2 = Stream.constant(s2, dtype="float")
        return BinOp(np.add, dtype="float")(s1, s2)
    return BinOp(np.add, dtype="float")(s1, s2).astype("float")


@Float.register(["radd", "__radd__"])
def radd(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the reversed addition of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from adding `s1` and `s2`.
    """
    return add(s1, s2)


@Float.register(["sub", "__sub__"])
def sub(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the subtraction of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from subtracting `s2` from `s1`.
    """
    if np.isscalar(s2):
        s2 = Stream.constant(s2, dtype="float")
        return BinOp(np.subtract, dtype="float")(s1, s2)
    return BinOp(np.subtract, dtype="float")(s1, s2)


@Float.register(["rsub", "__rsub__"])
def rsub(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the reverse subtraction of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from subtracting `s1` from `s2`.
    """
    if not np.isscalar(s2):
        raise Exception("Invalid stream operation.")
    s2 = Stream.constant(s2, dtype="float")
    return BinOp(np.subtract, dtype="float")(s2, s1)


@Float.register(["mul", "__mul__"])
def mul(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the multiplication of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from multiplying `s1` and `s2`.
    """
    if np.isscalar(s2):
        s2 = Stream.constant(s2, dtype="float")
        return BinOp(np.multiply, dtype="float")(s1, s2)
    return BinOp(np.multiply, dtype="float")(s1, s2)


@Float.register(["rmul", "__rmul__"])
def rmul(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the reverse multiplication of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from multiplying `s2` and `s1`.
    """
    return mul(s1, s2)


@Float.register(["div", "__truediv__"])
def truediv(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the division of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from dividing `s1` by `s2`.
    """
    if np.isscalar(s2):
        s2 = Stream.constant(s2, dtype="float")
        return BinOp(np.divide, dtype="float")(s1, s2)
    return BinOp(np.divide, dtype="float")(s1, s2)


@Float.register(["rdiv", "__rtruediv__"])
def rtruediv(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    """Computes the reverse division of two float streams.

    Parameters
    ----------
    s1 : `Stream[float]`
        The first float stream.
    s2 : `Stream[float]` or float
        The second float stream.

    Returns
    -------
    `Stream[float]`
        A stream created from dividing `s2` by `s1`.
    """
    if not np.isscalar(s2):
        raise Exception("Invalid stream operation.")
    s2 = Stream.constant(s2, dtype="float")
    return BinOp(np.divide, dtype="float")(s2, s1)


@Float.register(["abs", "__abs__"])
def abs(s: "Stream[float]") -> "Stream[float]":
    """Computes the absolute value of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The absolute value stream of `s`.
    """
    return s.apply(np.abs).astype("float")


@Float.register(["neg", "__neg__"])
def neg(s: "Stream[float]") -> "Stream[float]":
    """Computes the negation of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        The negated stream of `s`.
    """
    return s.apply(np.negative).astype("float")


@Float.register(["pow", "__pow__"])
def pow(s: "Stream[float]", power: float) -> "Stream[float]":
    """Computes the power of a float stream.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    power : float
        The power to raise `s` by.

    Returns
    -------
    `Stream[float]`
        The power stream of `s`.
    """
    return s.apply(lambda x: np.power(x, power)).astype("float")
