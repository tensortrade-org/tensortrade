
from tensortrade.base.core import Basic


class ExampleBasic(Basic):

    def __init__(self, msg):
        self.msg = msg


class Environment(Basic):

    def __init__(self):
        pass

    def step(self):
        self.clock.increment()


def test_basic_init():

    env = Environment()

    basic0 = ExampleBasic("Hello I'm basic 0!")
    assert basic0.created_at == 0


def test_basic_created_at():

    env = Environment()
    env.step()

    basic1 = ExampleBasic("Hello I'm basic 1!")
    assert basic1.created_at == 1

    for i in range(10):
        env.step()

    basic11 = ExampleBasic("Hello I'm basic 11!")
    assert basic11.created_at == 11