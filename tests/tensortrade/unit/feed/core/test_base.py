

from tensortrade.feed.core import Stream, NameSpace

from tensortrade.feed.core.base import Placeholder


class Counter(Stream):

    def __init__(self):
        super().__init__(dtype="string")
        self.count = None

    def forward(self):
        if self.count is None:
            self.count = 0
        else:
            self.count += 1
        return self.count

    def has_next(self):
        return True

    def reset(self):
        self.count = None


def test_stream():

    counter = Counter()

    assert counter.value is None
    assert counter.forward() == 0
    assert counter.forward() == 1

    counter.reset()

    assert counter.value is None
    assert counter.forward() == 0
    assert counter.forward() == 1


def test_stream_head():

    c1 = Counter().rename("c1")

    with NameSpace("world"):
        c2 = Counter().rename("c1")

    assert c1.name == "c1"
    assert c2.name == "world:/c1"



def test_stream_source():

    s = Stream.source(range(10))

    assert s.forward() == 0
    assert s.forward() == 1

    s.reset()

    assert s.forward() == 0
    assert s.forward() == 1


    def g():
        done = False
        i = 0
        yield i

        while not done:
            i += 1
            yield i

    s = Stream.source(g)

    assert s.forward() == 0
    assert s.forward() == 1

    s.reset()

    assert s.forward() == 0
    assert s.forward() == 1


def test_placholder():

    s = Stream.placeholder(dtype="float")

    s.push(5)

    assert s.value == 5
