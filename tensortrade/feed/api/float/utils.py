
import numpy as np

from tensortrade.feed import Stream
from tensortrade.feed.core.operators import BinOp

from tensortrade.feed import Float


@Float.register(["ceil"])
def ceil(s: "Stream[float]") -> "Stream[float]":
    return s.apply(np.ceil).astype("float")


@Float.register(["floor"])
def floor(s: "Stream[float]") -> "Stream[float]":
    return s.apply(np.floor).astype("float")


@Float.register(["sqrt"])
def sqrt(s: "Stream[float]") -> "Stream[float]":
    return s.apply(np.sqrt).astype("float")


@Float.register(["square"])
def square(s: "Stream[float]") -> "Stream[float]":
    return s.apply(np.square).astype("float")


@Float.register(["log"])
def log(s: "Stream[float]") -> "Stream[float]":
    return s.apply(np.log).astype("float")


@Float.register(["pct_change"])
def pct_change(s: "Stream[float]", periods: int = 1, fill_method: str = "pad") -> "Stream[float]":
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
    return BinOp(np.subtract)(s, s.lag(periods)).astype("float")
