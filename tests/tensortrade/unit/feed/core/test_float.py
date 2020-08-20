
from tensortrade.feed import Stream

from tests.utils.ops import assert_op


def test_float_accessor():

    s = Stream.source([1, 2, 3, 4, 5])

    w = s.float.square().rename("w")

    assert_op([w], [1, 4, 9, 16, 25])
