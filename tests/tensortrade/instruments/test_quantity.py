

from tensortrade.instruments import *


def test_quantity_initialization():
    q = Quantity(USD, 10000)

    assert q.instrument == USD
    assert q.amount == 10000

