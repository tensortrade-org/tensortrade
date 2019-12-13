
import pytest

from tensortrade.instruments import *
from tensortrade.base.exceptions import *


# Initialization
def test_valid_init():
    q = Quantity(BTC, 10000)

    assert q
    assert q.instrument == BTC
    assert q.size == 10000

    q = Quantity(BTC, 10000, path_id="f4cfeeae-a3e4-42e9-84b9-a24ccd2eebeb")

    assert q
    assert q.instrument == BTC
    assert q.size == 10000
    assert q.path_id == "f4cfeeae-a3e4-42e9-84b9-a24ccd2eebeb"


def test_invalid_init():

    with pytest.raises(InvalidNegativeQuantity):
        q = Quantity(BTC, -10000)

    with pytest.raises(TypeError):
        q = Quantity(BTC, "BTC")


# Locking
def test_locking():

    path_id = "f4cfeeae-a3e4-42e9-84b9-a24ccd2eebeb"

    q = Quantity(BTC, 10000)
    assert not q.is_locked

    q = Quantity(BTC, 10000, path_id=path_id)
    assert q.is_locked

    q = Quantity(BTC, 10000)
    q.path_id = path_id
    assert q.is_locked

    q = Quantity(BTC, 10000)
    q.lock_for(path_id)
    assert q.is_locked


# Addition
def test_valid_add():

    # Quantity
    q1 = Quantity(BTC, 10000)
    q2 = Quantity(BTC, 500)
    q = q1 + q2

    assert q.size == 10500
    assert q.instrument == BTC

    # int
    q1 = Quantity(BTC, 10000)
    q2 = 500
    q = q1 + q2

    # float
    assert q.size == 10500
    assert q.instrument == BTC

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

    # Quantity
    q1 = Quantity(BTC, 10000)
    q2 = "ETH"

    with pytest.raises(InvalidNonNumericQuantity):
        q = q1 + q2


# Iterative add
def test_valid_iadd():

    # Quantity
    q = Quantity(BTC, 10000)
    q += Quantity(BTC, 500)

    assert q.size == 10500
    assert q.instrument == BTC

    # int
    q = Quantity(BTC, 10000)
    q += 500

    # float
    assert q.size == 10500
    assert q.instrument == BTC

    q = Quantity(BTC, 10000)
    q += 500.0

    assert q.size == 10500
    assert q.instrument == BTC


def test_invalid_iadd():

    # Quantity
    q = Quantity(BTC, 10000)

    with pytest.raises(IncompatibleInstrumentOperation):
        q += Quantity(ETH, 500)

    q = Quantity(BTC, 10000)

    with pytest.raises(InvalidNonNumericQuantity):
        q += "ETH"


# Subtraction
def test_valid_subtraction():

    # Quantity
    q1 = Quantity(BTC, 1000)
    q2 = Quantity(BTC, 500)

    q = q1 - q2

    assert q.size == 500
    assert q.instrument == BTC

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


def test_invalid_subtraction():

    # Quantity
    q1 = Quantity(BTC, 500)
    q2 = Quantity(BTC, 1000)

    with pytest.raises(InvalidNegativeQuantity):
        q = q1 - q2

    q1 = Quantity(BTC, 500)
    q2 = Quantity(ETH,1000)

    with pytest.raises(IncompatibleInstrumentOperation):
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

    # int
    q = Quantity(BTC, 1000)
    q -= 500

    # float
    assert q.size == 500
    assert q.instrument == BTC

    q = Quantity(BTC, 1000)
    q -= 500.0

    assert q.size == 500
    assert q.instrument == BTC


def test_invalid_isub():

    # Quantity
    q = Quantity(BTC, 1000)

    with pytest.raises(IncompatibleInstrumentOperation):
        q -= Quantity(ETH, 500)

    with pytest.raises(InvalidNegativeQuantity):
        q -= Quantity(BTC, 1500)

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
def test_multiplication():

    # Quantity
    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    q = q1 * q2

    # int
    assert q.size == 250
    assert q.instrument == ETH

    q1 = Quantity(ETH, 50)
    q2 = 5
    q = q1 * q2

    # float
    assert q.size == 250
    assert q.instrument == ETH

    q1 = Quantity(ETH, 50)
    q2 = 5.0
    q = q1 * q2

    assert q.size == 250
    assert q.instrument == ETH


def test_invalid_multiplication():

    # Quantity
    q1 = Quantity(ETH, 5)
    q2 = Quantity(BTC, 50)

    with pytest.raises(IncompatibleInstrumentOperation):
        q = q1 * q2

    # Not a number
    q1 = Quantity(ETH, 5)
    q2 = "BTC"

    with pytest.raises(InvalidNonNumericQuantity):
        q = q1 * q2


# Division
def test_division():

    # Quantity
    q1 = Quantity(ETH, 50)
    q2 = Quantity(ETH, 5)
    q = q1 / q2

    assert q.size == 10
    assert q.instrument == ETH

    # int
    q1 = Quantity(ETH, 50)
    q2 = 5
    q = q1 / q2

    assert q.size == 10
    assert q.instrument == ETH

    # float
    q1 = Quantity(ETH, 50)
    q2 = 5.0
    q = q1 / q2

    assert q.size == 10
    assert q.instrument == ETH


def test_invalid_division():

    # Instruments do not match
    q1 = Quantity(ETH, 50)
    q2 = Quantity(BTC, 5)

    with pytest.raises(Exception):
        q = q1 / q2

    # Not a number
    q1 = Quantity(ETH, 50)
    q2 = "BTC"

    with pytest.raises(InvalidNonNumericQuantity):
        q = q1 / q2


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

    # Not a number
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

    # Not a number
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
        (q1 == q2) and (q2 == q1)


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


# Negation
def test_valid_negation():

    q = Quantity(ETH, 5)
    neg_q = -q
    assert neg_q == -5
