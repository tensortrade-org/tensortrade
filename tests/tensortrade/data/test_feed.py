

from typing import List

from tensortrade.data import DataFeed, DataSource


class Array(DataSource):

    def __init__(self, array: List[int]):
        super().__init__('array')
        self.cursor = 0
        self.array = array

    def next(self):
        v = {self.cursor: self.array[self.cursor]}
        self.cursor += 1
        return v

    def reset(self):
        self.cursor = 0


def test_init():
    sources = [
        Array([1, 2, 3]),
        Array([4, 5, 6]),
        Array([7, 8, 9])
    ]
    feed = DataFeed(sources=sources)

    assert feed
    assert feed.sources == sources


def test_next():
    sources = [
        Array([1, 2, 3]),
        Array([4, 5, 6]),
        Array([7, 8, 9])
    ]
    feed = DataFeed(sources=sources)

    assert isinstance(feed, DataSource)

    feed.next()