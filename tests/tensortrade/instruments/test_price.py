
import pytest

from tensortrade.instruments import Price, TradingPair, USD, BTC, ETH, Quantity
from tensortrade.base.exceptions import *
from numbers import Number


# Initialization
def test_valid_init():

    price = Price(7000, TradingPair(USD, BTC))
    assert price
    assert isinstance(price.rate, Number)
    assert price.rate == 7000

    assert isinstance(price.pair, TradingPair)
    assert price.pair == TradingPair(USD, BTC)

    price = Price(7000, USD/BTC)
    assert price
    assert isinstance(price.rate, Number)
    assert price.rate == 7000

    assert isinstance(price.pair, TradingPair)
    assert price.pair == USD / BTC


def test_invalid_init():

    with pytest.raises(InvalidNegativeQuantity):
        Price(-500, USD/BTC)


# Addition
def test_valid_add():
    # Price
    p1 = Price(10000, USD/BTC)
    p2 = Price(500, USD/BTC)
    p = p1 + p2
    assert p.rate == 10500
    assert p.pair == USD/BTC

    p = p2 + p1
    assert p.rate == 10500
    assert p.pair == USD / BTC

    # int
    p1 = Price(10000, USD / BTC)
    p2 = 500
    p = p1 + p2
    assert p.rate == 10500
    assert p.pair == USD / BTC

    p = p2 + p1
    assert p.rate == 10500
    assert p.pair == USD / BTC

    # float
    p1 = Price(10000, USD / BTC)
    p2 = 500.0
    p = p1 + p2
    assert p.rate == 10500
    assert p.pair == USD / BTC

    p = p2 + p1
    assert p.rate == 10500
    assert p.pair == USD / BTC


def test_invalid_add():

    # Price with different trading pairs
    p1 = Price(10000, USD/BTC)
    p2 = Price(500, BTC/USD)

    with pytest.raises(IncompatibleTradingPairOperation):
        p1 + p2

    # Not a number
    p1 = Price(500, USD / BTC)
    p2 = "ETH"
    with pytest.raises(InvalidNonNumericQuantity):
        p1 + p2


def test_valid_radd():

    # int
    p1 = 500
    p2 = Price(10000, USD / BTC)
    p = p1 + p2
    assert p.rate == 10500
    assert p.pair == USD / BTC

    # float
    p1 = 500.0
    p2 = Price(10000, USD / BTC)
    p = p1 + p2
    assert p.rate == 10500
    assert p.pair == USD / BTC


def test_invalid_radd():
    # Not a number
    p1 = "ETH"
    p2 = Price(500, USD / BTC)
    with pytest.raises(InvalidNonNumericQuantity):
        p1 + p2


# Iterative Addition
def test_valid_iadd():
    # Price
    p = Price(10000, USD / BTC)
    p += Price(500, USD / BTC)
    assert p.rate == 10500
    assert p.pair == USD / BTC

    # int
    p = Price(10000, USD / BTC)
    p += 500
    assert p.rate == 10500
    assert p.pair == USD / BTC

    # float
    p = Price(10000, USD / BTC)
    p += 500.0
    assert p.rate == 10500
    assert p.pair == USD / BTC


def test_invalid_iterative_add():

    # float
    p = Price(10000, USD / BTC)
    with pytest.raises(IncompatibleTradingPairOperation):
        p += Price(500, BTC / USD)

    # Not a number
    p = Price(500, USD / BTC)
    with pytest.raises(InvalidNonNumericQuantity):
        p += "ETH"


# Subtraction
def test_valid_sub():
    # Price
    p1 = Price(1000, USD / BTC)
    p2 = Price(500, USD / BTC)
    p = p1 - p2
    assert p.rate == 500
    assert p.pair == USD / BTC

    # int
    p1 = Price(1000, USD / BTC)
    p2 = 500
    p = p1 - p2
    assert p.rate == 500
    assert p.pair == USD / BTC

    # float
    p1 = Price(1000, USD / BTC)
    p2 = 500.0
    p = p1 - p2
    assert p.rate == 500
    assert p.pair == USD / BTC


def test_invalid_sub():

    # Price with negative difference
    p1 = Price(500, USD / BTC)
    p2 = Price(1000, USD / BTC)
    with pytest.raises(InvalidNegativeQuantity):
        p1 - p2

    # Price with different trading pairs
    p1 = Price(500, USD / BTC)
    p2 = Price(1000, BTC / USD)
    with pytest.raises(IncompatibleTradingPairOperation):
        p1 - p2

    # int
    p1 = Price(500, USD / BTC)
    p2 = 1000
    with pytest.raises(InvalidNegativeQuantity):
        p1 - p2

    # float
    p1 = Price(500, USD / BTC)
    p2 = 1000.0
    with pytest.raises(InvalidNegativeQuantity):
        p1 - p2

    # Not a number
    p1 = Price(500, USD / BTC)
    p2 = "ETH"
    with pytest.raises(InvalidNonNumericQuantity):
        p1 - p2


def test_valid_rsub():
    # Price
    p1 = Price(1000, USD / BTC)
    p2 = Price(500, USD / BTC)
    p = p1 - p2
    assert p.rate == 500
    assert p.pair == USD / BTC

    # int
    p1 = Price(1000, USD / BTC)
    p2 = 500
    p = p1 - p2
    assert p.rate == 500
    assert p.pair == USD / BTC

    # float
    p1 = Price(1000, USD / BTC)
    p2 = 500.0
    p = p1 - p2
    assert p.rate == 500
    assert p.pair == USD / BTC


def test_invalid_rsub():

    # int
    p1 = 500
    p2 = Price(1000, USD / BTC)
    with pytest.raises(InvalidNegativeQuantity):
        p1 - p2

    # float
    p1 = 500.0
    p2 = Price(1000, USD / BTC)
    with pytest.raises(InvalidNegativeQuantity):
        p1 - p2

    # Not a number
    p1 = "ETH"
    p2 = Price(1000, USD / BTC)
    with pytest.raises(InvalidNonNumericQuantity):
        p1 - p2


# Iterative Subtraction
def test_valid_isub():
    # Price
    p = Price(1000, USD / BTC)
    p -= Price(500, USD / BTC)
    assert p.rate == 500
    assert p.pair == USD / BTC

    # int
    p = Price(1000, USD / BTC)
    p -= 500
    assert p.rate == 500
    assert p.pair == USD / BTC

    # float
    p = Price(1000, USD / BTC)
    p -= 500.0
    assert p.rate == 500
    assert p.pair == USD / BTC


def test_invalid_isub():
    # Price with negative difference
    p = Price(500, USD / BTC)
    with pytest.raises(InvalidNegativeQuantity):
        p -= Price(1000, USD / BTC)

    # Price with different trading pairs
    p = Price(500, USD / BTC)
    with pytest.raises(IncompatibleTradingPairOperation):
        p -= Price(1000, BTC / USD)

    # int
    p = Price(500, USD / BTC)
    with pytest.raises(InvalidNegativeQuantity):
        p -= 1000

    # float
    p = Price(500, USD / BTC)
    with pytest.raises(InvalidNegativeQuantity):
        p -= 1000.0

    # Not a number
    p = Price(500, USD / BTC)
    with pytest.raises(InvalidNonNumericQuantity):
        p -= "ETH"


# Multiplication
def test_valid_mul():
    # Price
    p1 = Price(50, USD / BTC)
    p2 = Price(5, USD / BTC)
    p = p1 * p2
    assert p.rate == 250
    assert p.pair == USD / BTC

    # Quantity
    p = Price(50, USD / BTC)
    size = Quantity(BTC, 5)
    q = p * size
    assert isinstance(q, Quantity)
    assert q.size == 250
    assert q.instrument == USD

    # Quantity with Path ID
    p = Price(50, USD / BTC)
    size = Quantity(BTC, 5, path_id="fake_id")
    q = p * size
    assert isinstance(q, Quantity)
    assert q.size == 250
    assert q.instrument == USD
    assert q.path_id == "fake_id"

    # int
    p1 = Price(50, USD/BTC)
    p2 = 5
    p = p1 * p2
    assert p.rate == 250
    assert p.pair == USD / BTC

    # float
    p1 = Price(50, USD / BTC)
    p2 = 5.0
    p = p1 * p2
    assert p.rate == 250
    assert p.pair == USD / BTC


def test_invalid_mul():
    # Price with different trading pairs
    p1 = Price(50, USD / BTC)
    p2 = Price(50, BTC / USD)
    with pytest.raises(IncompatibleTradingPairOperation):
        p1 * p2

    # Quantity with instrument different than the quote
    p1 = Price(50, USD / BTC)
    p2 = Quantity(USD, 50)
    with pytest.raises(IncompatiblePriceQuantityOperation):
        p1 * p2


def test_valid_rmul():
    # Quantity
    size = Quantity(BTC, 5)
    p = Price(50, USD / BTC)
    q = size * p
    assert isinstance(q, Quantity)
    assert q.size == 250
    assert q.instrument == USD

    # Quantity with Path ID
    size = Quantity(BTC, 5, path_id="fake_id")
    p = Price(50, USD / BTC)
    q = size * p
    assert isinstance(q, Quantity)
    assert q.size == 250
    assert q.instrument == USD
    assert q.path_id == "fake_id"

    # int
    p1 = 5
    p2 = Price(50, USD / BTC)
    p = p1 * p2
    assert p.rate == 250
    assert p.pair == USD / BTC

    # float
    p1 = 5.0
    p2 = Price(50, USD / BTC)
    p = p1 * p2
    assert p.rate == 250
    assert p.pair == USD / BTC


def test_invalid_rmul():
    # Quantity with instrument different than the quote
    p1 = Quantity(USD, 50)
    p2 = Price(50, USD / BTC)
    with pytest.raises(IncompatiblePriceQuantityOperation):
        p1 * p2


# Division
def test_valid_truediv():

    # Price
    p1 = Price(50, USD / BTC)
    p2 = Price(5, USD / BTC)
    p = p1 / p2
    assert p == 10

    # Price with different quote instrument
    # but same base instrument
    p1 = Price(50, BTC / USD)
    p2 = Price(5, BTC / ETH)
    p = p1 / p2
    assert p.rate == 10
    assert p.pair == ETH / USD

    # Price with different base instrument
    # but same quote instrument
    p1 = Price(50, USD / BTC)
    p2 = Price(5, ETH / BTC)
    p = p1 / p2
    assert p.rate == 10
    assert p.pair == USD / ETH

    # int
    p1 = Price(50, USD / BTC)
    p2 = 5
    p = p1 / p2
    assert p.rate == 10
    assert p.pair == USD / BTC

    # float
    p1 = Price(50, USD / BTC)
    p2 = 5.0
    p = p1 / p2
    assert p.rate == 10
    assert p.pair == USD / BTC


def test_invalid_truediv():
    # Price with different trading pairs
    p1 = Price(50, USD / BTC)
    p2 = Price(50, BTC / USD)
    with pytest.raises(IncompatibleTradingPairOperation):
        p1 / p2

    # Quantity with instrument different than the quote
    p1 = Price(50, USD / BTC)
    p2 = Quantity(USD, 50)
    with pytest.raises(IncompatiblePriceQuantityOperation):
        p1 / p2


def test_valid_rtruediv():

    # Quantity
    size = Quantity(USD, 50)
    p = Price(5, USD / BTC)
    q = size / p
    assert isinstance(q, Quantity)
    assert q.size == 10
    assert q.instrument == BTC

    # Quantity with Path ID
    size = Quantity(USD, 50, "fake_id")
    p = Price(5, USD / BTC)
    q = size / p
    assert isinstance(q, Quantity)
    assert q.size == 10
    assert q.instrument == BTC
    assert q.path_id == "fake_id"

    # int
    p1 = 50
    p2 = Price(5, USD / BTC)
    p = p1 / p2
    assert p.rate == 10
    assert p.pair == BTC / USD

    # float
    p1 = 50.0
    p2 = Price(5, USD / BTC)
    p = p1 / p2
    assert p.rate == 10
    assert p.pair == BTC / USD


def test_invalid_rtruediv():

    # Quantity with instrument different than the quote
    p2 = Quantity(USD, 50)
    p1 = Price(50, BTC / USD)
    with pytest.raises(IncompatiblePriceQuantityOperation):
        p1 / p2


def test_str():
    p = Price(7020, USD / BTC)
    assert str(p) == "7020.00 USD/BTC"

    p = Price(0.0034, BTC / USD)
    assert str(p) == "0.00340000 BTC/USD"
