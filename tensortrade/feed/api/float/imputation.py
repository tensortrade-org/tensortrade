
from tensortrade.feed.api.generic.imputation import FillNa, ForwardFill
from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.float import Float


@Float.register(["fillna"])
def fillna(s: "Stream[float]", fill_value: float = 0.0) -> "Stream[float]":
    """Fill in missing values with a fill value.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    fill_value : float
        A value to fill in missing values with.

    Returns
    -------
    `Stream[float]`
        An imputed stream via padding.
    """
    return FillNa(fill_value=fill_value)(s).astype("float")


@Float.register(["ffill"])
def ffill(s: "Stream[float]") -> "Stream[float]":
    """Fill in missing values by forward filling.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.

    Returns
    -------
    `Stream[float]`
        An imputed stream via forward filling.
    """
    return ForwardFill()(s).astype("float")
