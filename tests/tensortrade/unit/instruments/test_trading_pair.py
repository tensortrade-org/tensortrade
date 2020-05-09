
import pytest

from tensortrade.instruments import TradingPair, BTC, USD
from tensortrade.base.exceptions import InvalidTradingPair


def test_valid_init():

    pair = TradingPair(USD,  BTC)
    assert pair
    assert pair.base == USD
    assert pair.quote == BTC


def test_invalid_init():

    with pytest.raises(InvalidTradingPair):
        pair = TradingPair(BTC, BTC)


def test_str():
    pair = TradingPair(USD, BTC)
    assert str(pair) == "USD/BTC"
