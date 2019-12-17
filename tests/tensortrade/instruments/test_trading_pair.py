

import pytest

from tensortrade.instruments import Instrument, TradingPair
from tensortrade.base.exceptions import InvalidTradingPair


def test_valid_init():

    USD = Instrument("USD", 8, "U.S. Dollar")
    BTC = Instrument("BTC", 8, "Bitcoin")

    pair = TradingPair(USD,  BTC)
    assert pair
    assert pair.base == USD
    assert pair.quote == BTC


def test_invalid_init():
    BTC = Instrument("BTC", 8, "Bitcoin")

    with pytest.raises(InvalidTradingPair):
        pair = TradingPair(BTC, BTC)


def test_str():
    USD = Instrument("USD", 8, "U.S. Dollar")
    BTC = Instrument("BTC", 8, "Bitcoin")

    pair = TradingPair(USD, BTC)
    assert str(pair) == "USD/BTC"
