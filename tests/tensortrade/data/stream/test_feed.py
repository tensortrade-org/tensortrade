
import operator
import pytest

from tensortrade.data import DataFeed, DataSource, Stream
from tensortrade.data.stream.transform import BinOp


def test_init():
    sources = [
        Stream('a1', [1, 2, 3]),
        Stream('a2', [4, 5, 6]),
        Stream('a3', [7, 8, 9])
    ]
    feed = DataFeed(sources)

    assert feed


def test_stream_adding():
    a1 = Stream('a1', [1, 2, 3])
    a2 = Stream('a2', [4, 5, 6])

    t1 = BinOp("a1+a2", operator.add)(a1, a2)

    feed = DataFeed([a1, a2, t1])

    output = feed.next()

    assert output == {'a1': 1, 'a2': 4, 'a1+a2': 5}


def test_multi_step_adding():

    a1 = Stream('a1', [1, 2, 3])
    a2 = Stream('a2', [4, 5, 6])

    t1 = BinOp('t1', operator.add)(a1, a2)
    t2 = BinOp('t2', operator.add)(t1, a2)

    feed = DataFeed([a1, a2, t1, t2])

    output = feed.next()
    assert output == {'a1': 1, 'a2': 4, 't1': 5, 't2': 9}

    feed = DataFeed([a1, a2, t2])

    output = feed.next()
    assert output == {'a1': 1, 'a2': 4, 't2': 9}
