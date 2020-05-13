
from tensortrade.data.feed.core import Stream
from tensortrade.data.feed.api.boolean import Boolean


@Boolean.register(["invert"])
def invert(s: "Stream[bool]") -> "Stream[bool]":
    return s.apply(lambda x: not x).astype("bool")
