

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


def test_reduce_sum():

    s1 = Stream.source([1, 2, 3, 4, 5, 6, 7], dtype="float").rename("s1")
    s2 = Stream.source([7, 6, 5, 4, 3, 2, 1], dtype="float").rename("s2")

    w = Stream.reduce([s1, s2]).sum().rename("w")

    expected = [8, 8, 8, 8, 8, 8, 8]

    assert_op([w], expected)


def test_reduce_min():

    s1 = Stream.source([1, 2, 3, 4, 5, 6, 7], dtype="float").rename("s1")
    s2 = Stream.source([-3, 6, 4, 2, 0, 4, 10], dtype="float").rename("s2")

    w = Stream.reduce([s1, s2]).min().rename("w")

    expected = [-3, 2, 3, 2, 0, 4, 7]

    assert_op([w], expected)


def test_reduce_max():

    s1 = Stream.source([1, 2, 3, 4, 5, 6, 7], dtype="float").rename("s1")
    s2 = Stream.source([-3, 6, 4, 2, 0, 4, 10], dtype="float").rename("s2")

    w = Stream.reduce([s1, s2]).max().rename("w")

    expected = [1, 6, 4, 4, 5, 6, 10]

    assert_op([w], expected)


def test_reduce_prod():

    s1 = Stream.source([1, 2, 3, 4, 5, 6, 7], dtype="float").rename("s1")
    s2 = Stream.source([3, 6, 4, 2, 0, 4, 10], dtype="float").rename("s2")

    w = Stream.reduce([s1, s2]).prod().rename("w")

    expected = [3, 12, 12, 8, 0, 24, 70]

    assert_op([w], expected)
