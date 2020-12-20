

from tensortrade.feed import Stream
from tensortrade.feed.core.feed import PushFeed


def test_init_push_feed():

    s1 = Stream.placeholder(dtype="float").rename("s1")
    s2 = Stream.placeholder(dtype="float").rename("s2")


    feed = PushFeed([s1, s2])

    assert isinstance(feed, PushFeed)


def test_push_one_datum():

    s1 = Stream.placeholder(dtype="float").rename("s1")
    s2 = Stream.placeholder(dtype="float").rename("s2")


    feed = PushFeed([s1, s2])

    assert isinstance(feed, PushFeed)

    output = feed.push({"s1": 1, "s2": 6})

    assert output == {"s1": 1, "s2": 6}

    m1 = s1.clamp_min(0)
    m2 = s2.clamp_max(0)

    feed = PushFeed([
        s1.rename("v1"),
        s2.rename("v2"),
        m1.rename("v3"),
        m2.rename("v4")
    ])

    arr1 = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    arr2 = [-5, 5, -4, 4, -3, 3, -2, 2, -1, 1]

    expected = {
        "v1": [1, -1, 2, -2, 3, -3, 4, -4, 5, -5],
        "v2": [-5, 5, -4, 4, -3, 3, -2, 2, -1, 1],
        "v3": [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
        "v4": [-5, 0, -4, 0, -3, 0, -2, 0, -1, 0]
    }

    for i in range(len(expected)):
        output = feed.push({
            "v1": arr1[i],
            "v2": arr2[i]
        })

        assert output == {
            "v1": expected["v1"][i],
            "v2": expected["v2"][i],
            "v3": expected["v3"][i],
            "v4": expected["v4"][i]
        }
