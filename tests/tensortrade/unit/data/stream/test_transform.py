

import operator

from tensortrade.data import Stream, DataFeed, Module, BinOp, Select


def test_namespace():

    a = Stream("a", [1, 2, 3])

    with Module("world") as world:
        a1 = Stream("a1", [4, 5, 6])
        a2 = Stream("a2", [7, 8, 9])

        with Module("sub-world") as sub_world:
            a3 = Stream("a3", [10, 11, 12])
            a4 = Stream("a4", [13, 14, 15])

            t3 = BinOp("t3", operator.add)(a2, a4)

    t1 = BinOp("t1", operator.mul)(a, t3)

    feed = DataFeed()(t1, world, sub_world)

    assert feed.next() == {
        "world:/a1": 4,
        "world:/a2": 7,
        "world:/sub-world:/a3": 10,
        "world:/sub-world:/a4": 13,
        "world:/sub-world:/t3": 20,
        "t1": 20
    }
    feed.reset()
    assert feed.next() == {
        "world:/a1": 4,
        "world:/a2": 7,
        "world:/sub-world:/a3": 10,
        "world:/sub-world:/a4": 13,
        "world:/sub-world:/t3": 20,
        "t1": 20
    }


def test_select():

    a3 = Stream("a3", [3, 2, 1])

    with Module("world") as a:
        a1 = Stream("a1", [7, 8, 9])
        a2 = Stream("a2", [3, 2, 1])

    t1 = BinOp("t1", operator.mul)(a1, a3)

    s = Select("world:/a1")(t1, a)
    feed = DataFeed()(s)

    assert feed.next() == {"world:/a1": 7}
