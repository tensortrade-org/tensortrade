
from tensortrade.data import DataFeed, DataSource, Array


def test_init():
    sources = [
        Array('a1', [1, 2, 3]),
        Array('a2', [4, 5, 6]),
        Array('a3', [7, 8, 9])
    ]
    feed = DataFeed(outputs=sources)

    assert feed


def test_next():
    sources = [
        Array('a1', [1, 2, 3]),
        Array('a2', [4, 5, 6]),
        Array('a3', [7, 8, 9])
    ]
    feed = DataFeed(inputs=sources)

    data = feed.next()

    assert data == {'a1': 1, 'a2': 4, 'a3': 7}

