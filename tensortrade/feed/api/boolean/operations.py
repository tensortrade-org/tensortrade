
from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.boolean import Boolean


@Boolean.register(["invert"])
def invert(s: "Stream[bool]") -> "Stream[bool]":
    """Inverts the truth value of the given stream.

    Parameters
    ----------
    s: `Stream[bool]`
        A boolean stream.

    Returns
    -------
    `Stream[bool]`
        An inverted stream of `s`.
    """
    return s.apply(lambda x: not x).astype("bool")
