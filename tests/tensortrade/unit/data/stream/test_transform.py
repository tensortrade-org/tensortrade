

import pytest
import numpy as np

from tensortrade.data import Stream, DataFeed, Module, BinOp, Select


@pytest.mark.skip(reason="Soon to be deprecated.")
def test_namespace():

    a = Stream([1, 2, 3]).rename("a")

    with Module("world") as world:
        a1 = Stream([4, 5, 6]).rename("a1")
        a2 = Stream([7, 8, 9]).rename("a2")

        with Module("sub-world") as sub_world:
            a3 = Stream([10, 11, 12]).rename("a3")
            a4 = Stream([13, 14, 15]).rename("a4")

            t3 = BinOp(np.add)(a2, a4).rename("t3")

    t1 = BinOp(np.multiply)(a, t3).rename("t1")

    feed = DataFeed([t1, world, sub_world])

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

    a3 = Stream([3, 2, 1]).rename("a3")

    with Module("world") as a:
        a1 = Stream([7, 8, 9]).rename("a1")
        a2 = Stream([3, 2, 1]).rename("a2")

    t1 = BinOp(np.multiply)(a1, a3).rename("t1")

    s = Select("world:/a1")(t1, a)
    feed = DataFeed([s])

    assert feed.next() == {"world:/a1": 7}
