

from tensortrade.data.feed.core.base import Observable, Stream, NameSpace


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


def test_node():

    counter = Counter()

    assert counter.value is None
    assert counter.forward() == 0
    assert counter.forward() == 1

    counter.reset()

    assert counter.value is None
    assert counter.forward() == 0
    assert counter.forward() == 1


def test_node_head():

    c1 = Counter().rename("c1")

    with NameSpace("world"):
        c2 = Counter().rename("c1")

    assert c1.name == "c1"
    assert c2.name == "world:/c1"
