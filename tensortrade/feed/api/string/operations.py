"""
operations.py contain functions for streaming string operations.
"""

from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.string import String


@String.register(["capitalize"])
def capitalize(s: "Stream[str]") -> "Stream[str]":
    """Computes the capitalization of a stream.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.

    Returns
    -------
    `Stream[str]`
        A capitalized string stream.
    """
    return s.apply(lambda x: x.capitalize()).astype("string")


@String.register(["upper"])
def upper(s: "Stream[str]") -> "Stream[str]":
    """Computes the uppercase of a string stream.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.

    Returns
    -------
    `Stream[str]`
        A uppercase string stream.
    """
    return s.apply(lambda x: x.upper()).astype("string")


@String.register(["lower"])
def lower(s: "Stream[str]") -> "Stream[str]":
    """Computes the lowercase of a string stream.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.

    Returns
    -------
    `Stream[str]`
        A lowercase string stream.
    """
    return s.apply(lambda x: x.lower()).astype("string")


@String.register(["slice"])
def slice(s: "Stream[str]", start: int, end: int) -> "Stream[str]":
    """Computes the substring of a string stream.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.
    start : int
        The start of the slice.
    end : int
        The end of the slice.

    Returns
    -------
    `Stream[str]`
        A substring stream.
    """
    return s.apply(lambda x: x[start:end]).astype("string")


@String.register(["cat"])
def cat(s: "Stream[str]", word: str) -> "Stream[str]":
    """Computes the concatenation of a stream with a word.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.
    word : str
        A word to concatenate with the `s`.

    Returns
    -------
    `Stream[str]`
        A concatenated string stream.
    """
    return s.apply(lambda x: x + word).astype("string")


@String.register(["startswith"])
def startswith(s: "Stream[str]", word: str) -> "Stream[bool]":
    """Computes the boolean stream of a string starting with a specific value.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.
    word : str
        A word that a string value can start with.

    Returns
    -------
    `Stream[bool]`
        A boolean stream.
    """
    return s.apply(lambda x: x.startswith(word)).astype("bool")


@String.register(["endswith"])
def endswith(s: "Stream[str]", word: str) -> "Stream[bool]":
    """Computes the boolean stream of a string ending with a specific value.

    Parameters
    ----------
    s : `Stream[str]`
        A string stream.
    word : str
        A word that a string value can end with.

    Returns
    -------
    `Stream[bool]`
        A boolean stream.
    """
    return s.apply(lambda x: x.endswith(word)).astype("bool")
