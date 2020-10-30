

from tensortrade.feed import Stream, DataFeed


def test_generic():

    s1 = Stream.source(["hello", "my", "name", "is"], dtype="string")
    s2 = Stream.source([1, 2, 3, 4, 5, 6])

    g1 = s1.apply(lambda x: x[0]).rename("g1")
    g2 = s2.lag().rename("g2")

    feed = DataFeed([g1, g2])
    feed.compile()

    feed.next()
    assert feed.next() == {"g1": "m", "g2": 1}
