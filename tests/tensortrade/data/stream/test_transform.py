

import operator

from tensortrade.data import Array, DataFeed
from tensortrade.data.stream.transform import BinOp, Select, Namespace


def test_namespace():

    a1 = Array("a1", [7, 8, 9])
    a2 = Array("a2", [3, 2, 1])

    t1 = BinOp("t1", operator.mul)(a1, a2)

    a = Namespace("world")(a1, a2)

    feed = DataFeed([t1, a])

    assert feed.next() == {"world:/a1": 7, "world:/a2": 3, "t1": 21}

    s = Select("world:/a1")(t1, a)
    feed = DataFeed([s])

    assert feed.next() == {"world:/a1": 7}

