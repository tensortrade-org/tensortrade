
import pytest

from tensortrade.oms.instruments import Instrument, Quantity
from tensortrade.core.exceptions import InvalidTradingPair


def test_init():
    BTC = Instrument("BTC", 8, "Bitcoin")
    assert BTC
    assert BTC.symbol == "BTC"
    assert BTC.precision == 8
    assert BTC.name == "Bitcoin"


# Equals
def test_valid_equals():
    BTC1 = Instrument("BTC", 8, "Bitcoin")
    BTC2 = Instrument("BTC", 8, "Bitcoin")
    assert BTC1 == BTC2

    BTC2 = Instrument("ETH", 8, "Bitcoin")
    assert not BTC1 == BTC2

    BTC2 = Instrument("BTC", 5, "Bitcoin")
    assert not BTC1 == BTC2

    BTC2 = Instrument("BTC", 8, "Etheruem")
    assert not BTC1 == BTC2


# Not equals
def test_not_equals():
    BTC1 = Instrument("BTC", 8, "Bitcoin")
    BTC2 = Instrument("BTC", 8, "Bitcoin")
    assert not BTC1 != BTC2

    BTC2 = Instrument("ETH", 8, "Bitcoin")
    assert BTC1 != BTC2

    BTC2 = Instrument("BTC", 5, "Bitcoin")
    assert BTC1 != BTC2

    BTC2 = Instrument("BTC", 8, "Etheruem")
    assert BTC1 != BTC2


# Right multiply
def test_valid_rmul():
    BTC = Instrument("BTC", 8, "Bitcoin")

    # int
    q = 8*BTC
    assert isinstance(q, Quantity)
    assert q.size == 8
    assert q.instrument == BTC

    # float
    q = 8.0*BTC
    assert isinstance(q, Quantity)
    assert q.size == 8.0
    assert q.instrument == BTC


def test_invalid_rmul():
    BTC = Instrument("BTC", 8, "Bitcoin")

    # int
    with pytest.raises(TypeError):
        q = BTC*8

    # float
    with pytest.raises(TypeError):
        q = BTC*8.0


# Division
def test_valid_truediv():
    BTC = Instrument("BTC", 8, "Bitcoin")
    ETH = Instrument("ETH", 8, "Etheruem")

    pair = BTC/ETH

    assert pair.base == BTC
    assert pair.quote == ETH


def test_invalid_truediv():
    BTC = Instrument("BTC", 8, "Bitcoin")

    with pytest.raises(InvalidTradingPair):
        pair = BTC / BTC


# String
def test_str():
    BTC = Instrument("BTC", 8, "Bitcoin")
    assert str(BTC) == "BTC"
