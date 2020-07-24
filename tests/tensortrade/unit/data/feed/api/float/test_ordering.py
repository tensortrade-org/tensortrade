
import pandas as pd

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


def test_clamp_min():
    array = [-1, 2, -3, 4, -5]

    s = Stream.source(array, dtype="float")

    w = s.clamp_min(0).rename("w")
    expected = list(pd.Series(array).apply(lambda x: x if x > 0 else 0))

    assert_op([w], expected)


def test_clamp_max():
    array = [-1, 2, -3, 4, -5]

    s = Stream.source(array, dtype="float")

    w = s.clamp_max(0).rename("w")
    expected = list(pd.Series(array).apply(lambda x: x if x < 0 else 0))

    assert_op([w], expected)


def test_min():

    s1 = Stream.source([-1, 2, -3, 4, -5], dtype="float")
    s2 = Stream.source([-1, 2, 3, 2, 1], dtype="float")

    w = s1.min(s2).rename("w")
    expected = [-1, 2, -3, 2, -5]

    assert_op([w], expected)


def test_max():

    s1 = Stream.source([-1, 2, -3, 4, -5], dtype="float")
    s2 = Stream.source([-1, 2, 3, 2, 1], dtype="float")

    w = s1.max(s2).rename("w")
    expected = [-1, 2, 3, 4, 1]

    assert_op([w], expected)
