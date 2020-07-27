
import numpy as np

from tensortrade.feed.core import DataFeed


def assert_op(streams, expected):

    feed = DataFeed(streams)
    feed.compile()

    actual = []
    while feed.has_next():
        d = feed.next()

        v = None
        for k in d.keys():
            if v is None:
                v = d[k]
            else:
                assert d[k] == v
        actual += [v]

    np.testing.assert_allclose(actual, expected)
