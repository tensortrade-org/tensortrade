

from tensortrade.instruments import *

import pytest

from tensortrade.instruments import *


def test_unit_init():
    BTC = Instrument("BTC", 8, "Bitcoin")
    assert BTC
    assert BTC.symbol == "BTC"
    assert BTC.precision == 8
    assert BTC.name == "Bitcoin"


def test_valid_addition():
    BTC1 = Instrument("BTC", 8, "Bitcoin")
    BTC2 = Instrument("BTC", 8, "Bitcoin")
    BTC = BTC1 + BTC2
    assert BTC.symbol == "BTC"
    assert BTC.precision == 8
    assert BTC.name == "Bitcoin"


def test_invalid_addition():
    m = Unit("m")
    s = Unit("s")
    with pytest.raises(Exception):
        m_plus_s = m + s


def test_valid_subtraction():
    m1 = Unit("m")
    m2 = Unit("m")
    m3 = m1 - m2
    assert m3.symbol == "m"


def test_invalid_subtraction():
    m = Unit("m")
    s = Unit("s")
    with pytest.raises(Exception):
        m_minus_s = m - s


def test_multiplication():
    m1 = Unit("m")
    m2 = Unit("s")
    m3 = m1*m2
    assert m3.symbol == "m*s"


def test_division():
    m1 = Unit("m")
    m2 = Unit("s")
    m3 = m1/m2
    assert m3.symbol == "m/s"


def test_make_quantity_with_unit():

    m = Unit("m")
    s = Unit("s")

    q = 5*m

    assert isinstance(q, Quantity)
    assert q.size == 5
    assert q.unit == m

    q = 5 * (m / s)

    assert isinstance(q, Quantity)
    assert q.size == 5
    assert q.unit == m / s