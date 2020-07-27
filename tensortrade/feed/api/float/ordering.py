
import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.core.operators import BinOp

from tensortrade.feed.api.float import Float


@Float.register(["clamp_min"])
def clamp_min(s: "Stream[float]", c_min: float) -> "Stream[float]":
    return BinOp(np.maximum)(s, Stream.constant(c_min)).astype("float")


@Float.register(["clamp_max"])
def clamp_max(s: "Stream[float]", c_max: float) -> "Stream[float]":
    return BinOp(np.minimum)(s, Stream.constant(c_max)).astype("float")


@Float.register(["clamp"])
def clamp(s: "Stream[float]", c_min: float, c_max: float) -> "Stream[float]":
    stream = s.clamp_min(c_min).astype("float")
    stream = stream.clamp_max(c_max).astype("float")
    return stream


@Float.register(["min"])
def min(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    return BinOp(np.minimum)(s1, s2).astype("float")


@Float.register(["max"])
def max(s1: "Stream[float]", s2: "Stream[float]") -> "Stream[float]":
    return BinOp(np.maximum)(s1, s2).astype("float")
