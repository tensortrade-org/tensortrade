
import pytest
import unittest.mock as mock

from tensortrade.oms.instruments import *
from tensortrade.core.exceptions import *

path_id = "f4cfeeae-a3e4-42e9-84b9-a24ccd2eebeb"
other_id = "7f3de243-0474-48d9-bf44-ca55ae07a70e"


# Initialization
def test_valid_init():

    # Quantity
    q = Quantity(BTC, 10000)
    assert q
    assert q.instrument == BTC
    assert q.size == 10000

    # Quantity with Path ID
    q = Quantity(BTC, 10000, path_id=path_id)
    assert q
    assert q.instrument == BTC
    assert q.size == 10000
    assert q.path_id == path_id


def test_invalid_init():

    with pytest.raises(InvalidNegativeQuantity):
        q = Quantity(BTC, -10000)

    with pytest.raises(TypeError):
        q = Quantity(BTC, "BTC")


# Locking
def test_locking():

    q = Quantity(BTC, 10000)
    assert not q.is_locked

    q = Quantity(BTC, 10000, path_id=path_id)
    assert q.is_locked

    q = Quantity(BTC, 10000)
    q.path_id = path_id
    assert q.is_locked

    q = Quantity(BTC, 10000).lock_for(path_id)
    assert q.is_locked


# Addition
def test_valid_add():

    # Quantity
    q1 = Quantity(BTC, 10000)
    q2 = Quantity(BTC, 500)
    q = q1 + q2
    assert q.size == 10500
    assert q.instrument == BTC

    # Quantity with Path ID
    q1 = Quantity(BTC, 10000, path_id=path_id)
    q2 = Quantity(BTC, 500)
    q = q1 + q2
    assert q.size == 10500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # Quantity with matching Path ID
    q1 = Quantity(BTC, 10000, path_id=path_id)
    q2 = Quantity(BTC, 500, path_id=path_id)
    q = q1 + q2
    assert q.size == 10500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # int
    q1 = Quantity(BTC, 10000)
    q2 = 500
    q = q1 + q2
    assert q.size == 10500
    assert q.instrument == BTC

    # float
    q1 = Quantity(BTC, 10000)
    q2 = 500.0
    q = q1 + q2
    assert q.size == 10500
    assert q.instrument == BTC


def test_invalid_add():

    # Quantity
    q1 = Quantity(BTC, 10000)
    q2 = Quantity(ETH, 500)
    with pytest.raises(IncompatibleInstrumentOperation):
        q = q1 + q2

    # Quantity with different Path IDs
    q1 = Quantity(BTC, 10000, path_id=path_id)
    q2 = Quantity(BTC, 500, path_id=other_id)
    with pytest.raises(QuantityOpPathMismatch):
        q = q1 + q2

    # str
    q1 = Quantity(BTC, 10000)
    q2 = "ETH"
    with pytest.raises(InvalidNonNumericQuantity):
        q = q1 + q2


# Iterative Addition
def test_valid_iadd():

    # Quantity
    q = Quantity(BTC, 10000)
    q += Quantity(BTC, 500)
    assert q.size == 10500
    assert q.instrument == BTC

    # Quantity with Path ID
    q = Quantity(BTC, 10000, path_id=path_id)
    q += Quantity(BTC, 500)
    assert q.size == 10500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # Quantity with matching Path ID
    q = Quantity(BTC, 10000, path_id=path_id)
    q += Quantity(BTC, 500, path_id=path_id)
    assert q.size == 10500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # int
    q = Quantity(BTC, 10000)
    q += 500
    assert q.size == 10500
    assert q.instrument == BTC

    # float
    q = Quantity(BTC, 10000)
    q += 500.0
    assert q.size == 10500
    assert q.instrument == BTC


def test_invalid_iadd():

    # Quantity
    q = Quantity(BTC, 10000)
    with pytest.raises(IncompatibleInstrumentOperation):
        q += Quantity(ETH, 500)

    # Quantity with different Path IDs
    q = Quantity(BTC, 10000, path_id=path_id)
    with pytest.raises(QuantityOpPathMismatch):
        q += Quantity(BTC, 500, path_id=other_id)

    # str
    q = Quantity(BTC, 10000)
    with pytest.raises(InvalidNonNumericQuantity):
        q += "ETH"


# Subtraction
def test_valid_sub():

    # Quantity
    q1 = Quantity(BTC, 1000)
    q2 = Quantity(BTC, 500)
    q = q1 - q2
    assert q.size == 500
    assert q.instrument == BTC

    # Quantity with Path ID
    q1 = Quantity(BTC, 1000, path_id=path_id)
    q2 = Quantity(BTC, 500)
    q = q1 - q2
    assert q.size == 500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # Quantity with matching Path ID
    q1 = Quantity(BTC, 1000, path_id=path_id)
    q2 = Quantity(BTC, 500, path_id=path_id)
    q = q1 - q2
    assert q.size == 500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # int
    q1 = Quantity(BTC, 1000)
    q2 = 500
    q = q1 - q2
    assert q.size == 500
    assert q.instrument == BTC

    # float
    q1 = Quantity(BTC, 1000)
    q2 = 500.0
    q = q1 - q2
    assert q.size == 500
    assert q.instrument == BTC


def test_invalid_sub():

    # Quantity with negative difference
    q1 = Quantity(BTC, 500)
    q2 = Quantity(BTC, 1000)
    with pytest.raises(InvalidNegativeQuantity):
        q = q1 - q2

    # Quantity with different instruments
    q1 = Quantity(BTC, 500)
    q2 = Quantity(ETH, 1000)
    with pytest.raises(IncompatibleInstrumentOperation):
        q = q1 - q2

    # Quantity with different Path IDs
    q1 = Quantity(BTC, 1000, path_id=path_id)
    q2 = Quantity(BTC, 500, path_id=other_id)
    with pytest.raises(QuantityOpPathMismatch):
        q = q1 - q2

    # int
    q1 = Quantity(BTC, 500)
    q2 = 1000
    with pytest.raises(InvalidNegativeQuantity):
        q = q1 - q2

    # float
    q1 = Quantity(BTC, 500)
    q2 = 1000.0
    with pytest.raises(InvalidNegativeQuantity):
        q = q1 - q2

    # Not a number
    q1 = Quantity(BTC, 500)
    q2 = "ETH"
    with pytest.raises(InvalidNonNumericQuantity):
        q = q1 - q2


# Iterative subtraction
def test_valid_isub():

    # Quantity
    q = Quantity(BTC, 1000)
    q -= Quantity(BTC, 500)
    assert q.size == 500
    assert q.instrument == BTC

    # Quantity with Path ID
    q = Quantity(BTC, 1000, path_id=path_id)
    q -= Quantity(BTC, 500)
    assert q.size == 500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # Quantity with matching Path ID
    q = Quantity(BTC, 1000, path_id=path_id)
    q -= Quantity(BTC, 500, path_id=path_id)
    assert q.size == 500
    assert q.instrument == BTC
    assert q.path_id == path_id

    # int
    q = Quantity(BTC, 1000)
    q -= 500
    assert q.size == 500
    assert q.instrument == BTC

    # float
    q = Quantity(BTC, 1000)
    q -= 500.0
    assert q.size == 500
    assert q.instrument == BTC


def test_invalid_isub():

    # Quantity with negative difference
    q = Quantity(BTC, 1000)
    with pytest.raises(InvalidNegativeQuantity):
        q -= Quantity(BTC, 1500)

    # Quantity with different instruments
    q = Quantity(BTC, 1000)
    with pytest.raises(IncompatibleInstrumentOperation):
        q -= Quantity(ETH, 500)

    # Quantity with different Path IDs
    q = Quantity(BTC, 1000, path_id=path_id)
    with pytest.raises(QuantityOpPathMismatch):
        q -= Quantity(BTC, 500, path_id=other_id)

    # int
    q = Quantity(BTC, 1000)
    with pytest.raises(InvalidNegativeQuantity):
        q -= 1500

    # float
    q = Quantity(BTC, 1000)
    with pytest.raises(InvalidNegativeQuantity):
        q -= 1500

    # Not a number
    q = Quantity(ETH, 5)
    with pytest.raises(InvalidNonNumericQuantity):
        q -= "BTC"


# Multiplication
def test_valid_mul():

    # Quantity
    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    q = q1 * q2
    assert q.size == 250
    assert q.instrument == ETH

    # Quantity with Path ID
    q1 = Quantity(BTC, 50, path_id=path_id)
    q2 = Quantity(BTC, 5)
    q = q1 * q2
    assert q.size == 250
    assert q.instrument == BTC
    assert q.path_id == path_id

    # Quantity with matching Path ID
    q1 = Quantity(BTC, 50, path_id=path_id)
    q2 = Quantity(BTC, 5, path_id=path_id)
    q = q1 * q2
    assert q.size == 250
    assert q.instrument == BTC
    assert q.path_id == path_id

    # int
    q1 = Quantity(ETH, 50)
    q2 = 5
    q = q1 * q2
    assert q.size == 250
    assert q.instrument == ETH

    # float
    q1 = Quantity(ETH, 50)
    q2 = 5.0
    q = q1 * q2
    assert q.size == 250
    assert q.instrument == ETH


def test_valid_rmul():

    # Quantity
    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    q = q2 * q1
    assert q.size == 250
    assert q.instrument == ETH

    # Quantity with Path ID
    q1 = Quantity(BTC, 50, path_id=path_id)
    q2 = Quantity(BTC, 5)
    q = q2 * q1
    assert q.size == 250
    assert q.instrument == BTC
    assert q.path_id == path_id

    # Quantity with matching Path ID
    q1 = Quantity(BTC, 50, path_id=path_id)
    q2 = Quantity(BTC, 5, path_id=path_id)
    q = q2 * q1
    assert q.size == 250
    assert q.instrument == BTC
    assert q.path_id == path_id

    # int
    q1 = Quantity(ETH, 50)
    q2 = 5
    q = q2 * q1
    assert q.size == 250
    assert q.instrument == ETH

    # float
    q1 = Quantity(ETH, 50)
    q2 = 5.0
    q = q2 * q1
    assert q.size == 250
    assert q.instrument == ETH


def test_invalid_mul():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(BTC, 50)
    with pytest.raises(IncompatibleInstrumentOperation):
        q = q1 * q2

    # Quantity with different Path IDs
    q1 = Quantity(BTC, 5, path_id=path_id)
    q2 = Quantity(BTC, 50, path_id=other_id)
    with pytest.raises(QuantityOpPathMismatch):
        q = q1 * q2

    # str
    q1 = Quantity(ETH, 5)
    q2 = "BTC"
    with pytest.raises(InvalidNonNumericQuantity):
        q = q1 * q2


# Less than
def test_valid_lt():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(ETH, 50)
    assert (q1 < q2) and not (q2 < q1)

    # int
    q1 = Quantity(ETH, 5)
    q2 = 50
    assert (q1 < q2) and not (q2 < q1)

    # float
    q1 = Quantity(ETH, 5)
    q2 = 50.0
    assert (q1 < q2) and not (q2 < q1)


def test_invalid_lt():

    # Quantity
    q1 = Quantity(BTC, 5)
    q2 = Quantity(ETH, 50)
    with pytest.raises(IncompatibleInstrumentOperation):
        assert (q1 < q2) and not (q2 < q1)

    # str
    q1 = Quantity(BTC, 5)
    q2 = "ETH"
    with pytest.raises(InvalidNonNumericQuantity):
        assert (q1 < q2)


# Greater than
def test_valid_gt():

    # Quantity
    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    assert (q1 > q2) and not (q2 > q1)

    # int
    q1 = Quantity(ETH, 50)
    q2 = 5
    assert (q1 > q2) and not (q2 > q1)

    # float
    q1 = Quantity(ETH, 50)
    q2 = 5.0
    assert (q1 > q2) and not (q2 > q1)


def test_invalid_gt():

    # Quantity
    q1 = Quantity(BTC, 50)
    q2 = Quantity(ETH, 5)
    with pytest.raises(IncompatibleInstrumentOperation):
        assert (q1 > q2) and not (q2 > q1)

    # str
    q1 = Quantity(BTC, 50)
    q2 = "ETH"
    with pytest.raises(InvalidNonNumericQuantity):
        assert (q1 > q2)


# Equals
def test_valid_equals():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(ETH, 5)
    assert (q1 == q2) and (q2 == q1)

    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    assert not (q1 == q2) and not (q2 == q1)

    # int
    q1 = Quantity(ETH, 5)
    q2 = 5
    assert (q1 == q2) and (q2 == q1)

    q1 = Quantity(ETH, 50)
    q2 = 5
    assert not (q1 == q2) and not (q2 == q1)

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = 5.0
    assert (q1 == q2) and (q2 == q1)

    q1 = Quantity(ETH, 50)
    q2 = 5.0
    assert not (q1 == q2) and not (q2 == q1)


def test_invalid_equals():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(BTC, 5)
    with pytest.raises(IncompatibleInstrumentOperation):
        assert (q1 == q2) and (q2 == q1)


# Not equals
def test_valid_unequals():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(ETH, 5)
    assert not (q1 != q2) and not (q2 != q1)

    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    assert (q1 != q2) and (q2 != q1)

    # int
    q1 = Quantity(ETH, 5)
    q2 = 5
    assert not (q1 != q2) and not (q2 != q1)

    q1 = Quantity(ETH, 50)
    q2 = 5
    assert (q1 != q2) and (q2 != q1)

    # float
    q1 = Quantity(ETH, 5)
    q2 = 5.0
    assert not (q1 != q2) and not (q2 != q1)

    q1 = Quantity(ETH, 50)
    q2 = 5.0
    assert (q1 != q2) and (q2 != q1)


def test_invalid_unequals():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(BTC, 5)
    with pytest.raises(IncompatibleInstrumentOperation):
        assert not (q1 != q2) and not (q2 != q1)


# Negation
def test_valid_negation():

    q = Quantity(ETH, 5)
    neg_q = -q
    assert neg_q == -5


def test_free():

    q = Quantity(ETH, 5)
    free = q.free()

    assert isinstance(free, Quantity)
    assert free.size == 5 and free.instrument == ETH
    assert not free.is_locked

    q = Quantity(ETH, 5, path_id="fake_id")
    free = q.free()

    assert isinstance(free, Quantity)
    assert free.size == 5 and free.instrument == ETH
    assert not free.is_locked


@mock.patch("tensortrade.instruments.ExchangePair")
def test_convert(mock_exchange_pair):

    exchange_pair = mock_exchange_pair.return_value
    exchange_pair.pair = USD/BTC
    exchange_pair.price = 9000

    # Test converts to Quote
    quantity = Quantity(USD, 1000)
    converted = quantity.convert(exchange_pair)

    assert float(converted.size) == 1000 / 9000
    assert converted.instrument == BTC

    # Test converts to Quote
    quantity = Quantity(BTC, 1.6)
    converted = quantity.convert(exchange_pair)

    assert float(converted.size) == 1.6 * 9000
    assert converted.instrument == USD
