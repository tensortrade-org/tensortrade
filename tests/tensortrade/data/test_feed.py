
from tensortrade.data import DataFeed, DataSource, Array


def test_init():
    sources = [
        Array([1, 2, 3]),
        Array([4, 5, 6]),
        Array([7, 8, 9])
    ]
    feed = DataFeed(sources=sources)

    assert feed


def test_next():
    sources = [
        Array([1, 2, 3]),
        Array([4, 5, 6]),
        Array([7, 8, 9])
    ]
    feed = DataFeed(sources=sources)

    assert isinstance(feed, DataSource)

    feed.next()