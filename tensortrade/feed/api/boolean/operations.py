
from tensortrade.feed import Stream
from tensortrade.feed import Boolean


@Boolean.register(["invert"])
def invert(s: "Stream[bool]") -> "Stream[bool]":
    return s.apply(lambda x: not x).astype("bool")
