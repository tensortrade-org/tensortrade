
from tensortrade.feed.api.generic.imputation import FillNa, ForwardFill
from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.float import Float


@Float.register(["fillna"])
def fillna(s: "Stream[float]", fill_value: float = 0.0) -> "Stream[float]":
    return FillNa(fill_value=fill_value)(s).astype("float")


@Float.register(["ffill"])
def ffill(s: "Stream[float]") -> "Stream[float]":
    return ForwardFill()(s).astype("float")
