from tensortrade.feed import DataFeed, Stream


def test_str_accessor():

    s = Stream.source(["hello", "my", "name", "i", "am", "the", "data", "feed"])

    w1 = s.str.upper().rename("w1")
    w2 = s.str.lower().rename("w2")
    w3 = w1.str.endswith("E").rename("w3")

    feed = DataFeed([w1, w2, w3])
    feed.compile()

    assert feed.next() == {"w1": "HELLO", "w2": "hello", "w3": False}


def test_bool_accessor():
    s = Stream.source(["hello", "my", "name", "i", "am", "the", "data", "feed"])

    w1 = s.str.upper().rename("w1")
    w2 = s.str.lower().rename("w2")
    w3 = w1.str.endswith("E").bool.invert().rename("w3")

    feed = DataFeed([w1, w2, w3])
    feed.compile()

    assert feed.next() == {"w1": "HELLO", "w2": "hello", "w3": True}


def test_str_methods():

    s = Stream.source(
        ["hello", "my", "name", "i", "am", "the", "data", "feed"], dtype="string"
    )

    w1 = s.upper().rename("w1")
    w2 = s.lower().rename("w2")
    w3 = w1.endswith("E").rename("w3")

    feed = DataFeed([w1, w2, w3])
    feed.compile()

    assert feed.next() == {"w1": "HELLO", "w2": "hello", "w3": False}


def test_bool_methods():
    s = Stream.source(
        ["hello", "my", "name", "i", "am", "the", "data", "feed"], dtype="string"
    )

    w1 = s.upper().rename("w1")
    w2 = s.lower().rename("w2")
    w3 = w1.endswith("E").invert().rename("w3")

    feed = DataFeed([w1, w2, w3])
    feed.compile()

    assert feed.next() == {"w1": "HELLO", "w2": "hello", "w3": True}
