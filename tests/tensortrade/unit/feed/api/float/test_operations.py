
import pandas as pd

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


def test_add():

    # (left, right) : (Stream, Stream)
    s1 = Stream.source([3, -4, 6, -7, 2, -6], dtype="float")
    s2 = Stream.source([-3, 4, -6, 7, -2, 6], dtype="float")

    w1 = s1.add(s2).rename("w1")
    w2 = (s1 + s2).rename("w2")

    assert_op([w1, w2], 6*[0])

    # (left, right) : (Stream, float)
    s1 = Stream.source([1, 2, 3, 4, 5, 6], dtype="float")
    s2 = 1

    w1 = s1.add(s2).rename("w1")
    w2 = (s1 + s2).rename("w2")

    assert_op([w1, w2], [2, 3, 4, 5, 6, 7])


def test_radd():
    # (left, right) : (float, Stream)
    s1 = 1
    s2 = Stream.source([1, 2, 3, 4, 5, 6], dtype="float")

    w = (s1 + s2).rename("w")

    assert_op([w], [2, 3, 4, 5, 6, 7])


def test_sub():
    expected = [0, 1, 2, 3, 4, 5]

    # (left, right) : (Stream, Stream)
    s1 = Stream.source([1, 2, 3, 4, 5, 6], dtype="float")
    s2 = Stream.source([1, 1, 1, 1, 1, 1], dtype="float")

    w1 = s1.sub(s2).rename("w1")
    w2 = (s1 - s2).rename("w2")

    assert_op([w1, w2], expected)

    # (left, right) : (Stream, float)
    w1 = s1.sub(1).rename("w1")
    w2 = (s1 - 1).rename("w2")

    assert_op([w1, w2], expected)


def test_rsub():
    # (left, right) : (float, Stream)
    s1 = 6
    s2 = Stream.source([1, 2, 3, 4, 5, 6], dtype="float")

    w = (s1 - s2).rename("w")

    assert_op([w], [5, 4, 3, 2, 1, 0])


def test_mul():
    expected = [2, 4, 6, 8, 10, 12]

    # (left, right) : (Stream, Stream)
    s1 = Stream.source([1, 2, 3, 4, 5, 6], dtype="float")
    s2 = Stream.source([2, 2, 2, 2, 2, 2], dtype="float")

    w1 = s1.mul(s2).rename("w1")
    w2 = (s1 * s2).rename("w2")

    assert_op([w1, w2], expected)

    # (left, right) : (Stream, float)
    w1 = s1.mul(2).rename("w1")
    w2 = (s1 * 2).rename("w2")

    assert_op([w1, w2], expected)


def test_rmul():
    expected = [2, 4, 6, 8, 10, 12]

    # (left, right) : (Stream, Stream)
    s = Stream.source([1, 2, 3, 4, 5, 6], dtype="float")

    # (left, right) : (Stream, float)
    w = (2 * s).rename("w")

    assert_op([w], expected)


def test_div():
    expected = [1, 2, 3, 4, 5, 6]

    # (left, right) : (Stream, Stream)
    s1 = Stream.source([2, 4, 6, 8, 10, 12], dtype="float")
    s2 = Stream.source([2, 2, 2, 2, 2, 2], dtype="float")

    w1 = s1.div(s2).rename("w1")
    w2 = (s1 / s2).rename("w2")

    assert_op([w1, w2], expected)

    # (left, right) : (Stream, float)
    w1 = s1.div(2).rename("w1")
    w2 = (s1 / 2).rename("w2")

    assert_op([w1, w2], expected)


def test_rdiv():
    expected = [6, 3, 2, 3/2, 6/5, 1]

    # (left, right) : (Stream, Stream)
    s = Stream.source([2, 4, 6, 8, 10, 12], dtype="float")

    # (left, right) : (Stream, float)
    w = (12 / s).rename("w")

    assert_op([w], expected)


def test_abs():
    s = Stream.source([3, -4, 6, -7, 2, -6], dtype="float")

    s1 = s.abs().rename("s1")
    s2 = abs(s).rename("s2")

    assert_op([s1, s2], [3, 4, 6, 7, 2, 6])


def test_neg():
    s = Stream.source([3, -4, 6, -7, 2, -6], dtype="float")

    s1 = s.neg().rename("s1")
    s2 = (-s).rename("s2")

    assert_op([s1, s2], [-3, 4, -6, 7, -2, 6])


def test_pow():
    array = [1, -2, 3, -4, 5, -6]

    s = Stream.source(array, dtype="float")

    s1 = s.pow(3).rename("s1")
    s2 = (s**3).rename("s2")

    expected = list(pd.Series(array)**3)

    assert_op([s1, s2], expected)
