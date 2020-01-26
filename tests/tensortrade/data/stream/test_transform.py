

import operator

from tensortrade.data import Stream, DataFeed
from tensortrade.data.stream.transform import BinOp, Select, Namespace


def test_namespace():

    a1 = Stream("a1", [7, 8, 9])
    a2 = Stream("a2", [3, 2, 1])

    t1 = BinOp("t1", operator.mul)(a1, a2)

    a = Namespace("world")(a1, a2)

    feed = DataFeed([t1, a])

    assert feed.next() == {"world:/a1": 7, "world:/a2": 3, "t1": 21}
    feed.reset()
    assert feed.next() == {"world:/a1": 7, "world:/a2": 3, "t1": 21}


def test_select():
    a1 = Stream("a1", [7, 8, 9])
    a2 = Stream("a2", [3, 2, 1])

    t1 = BinOp("t1", operator.mul)(a1, a2)
    a = Namespace("world")(a1, a2)

    s = Select("world:/a1")(t1, a)
    feed = DataFeed([s])

    print(a1.name, a1.inbound, a1.outbound)
    print(a2.name, a2.inbound, a2.outbound)
    print(t1.name, t1.inbound, t1.outbound)
    print(a.name, a.inbound, a.outbound)
    print(s.name, s.inbound, s.outbound)
    print(feed.inputs)

    assert feed.next() == {"world:/a1": 7}