

from tensortrade.data.feed.core.base import Stream
from tensortrade.data.feed.api.string import String


@String.register(["capitalize"])
def capitalize(s: "Stream[str]") -> "Stream[str]":
    return s.apply(lambda x: x.capitalize()).astype("string")


@String.register(["upper"])
def upper(s: "Stream[str]") -> "Stream[str]":
    return s.apply(lambda x: x.upper()).astype("string")


@String.register(["lower"])
def lower(s: "Stream[str]") -> "Stream[str]":
    return s.apply(lambda x: x.lower()).astype("string")


@String.register(["slice"])
def slice(s: "Stream[str]", start: int, end: int) -> "Stream[str]":
    return s.apply(lambda x: x[start:end]).astype("string")


@String.register(["cat"])
def cat(s: "Stream[str]", word: str) -> "Stream[str]":
    return s.apply(lambda x: x + word).astype("string")


@String.register(["startswith"])
def startswith(s: "Stream[str]", word: str) -> "Stream[bool]":
    return s.apply(lambda x: x.startswith(word)).astype("bool")


@String.register(["endswith"])
def endswith(s: "Stream[str]", word: str) -> "Stream[bool]":
    return s.apply(lambda x: x.endswith(word)).astype("bool")
