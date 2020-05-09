
import numpy as np

from tensortrade.data import DataFeed, Stream, BinOp


def test_init():
    sources = [
        Stream([1, 2, 3]).rename("a1"),
        Stream([4, 5, 6]).rename("a2"),
        Stream([7, 8, 9]).rename("a3")
    ]
    feed = DataFeed(sources)

    assert feed


def test_stream_adding():
    a1 = Stream([1, 2, 3]).rename("a1")
    a2 = Stream([4, 5, 6]).rename("a2")

    t1 = BinOp(np.add)(a1, a2).rename("a1+a2")

    feed = DataFeed([t1, a1, a2])

    output = feed.next()

    assert output == {'a1': 1, 'a2': 4, 'a1+a2': 5}


def test_multi_step_adding():

    a1 = Stream([1, 2, 3]).rename("a1")
    a2 = Stream([4, 5, 6]).rename("a2")

    t1 = BinOp(np.add)(a1, a2).rename("t1")
    t2 = BinOp(np.add)(t1, a2).rename("t2")

    feed = DataFeed([a1, a2, t1, t2])

    output = feed.next()
    assert output == {'a1': 1, 'a2': 4, 't1': 5, 't2': 9}

    feed = DataFeed([a1, a2, t2])

    output = feed.next()
    assert output == {'a1': 1, 'a2': 4, 't2': 9}
