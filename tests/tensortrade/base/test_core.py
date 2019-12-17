
from tensortrade.base.core import TimedIdentifiable


class ExampleTimedIdentifiable(TimedIdentifiable):

    def __init__(self, msg):
        self.msg = msg


class Environment(TimedIdentifiable):

    def __init__(self):
        pass

    def step(self):
        self.clock.increment()


def test_basic_init():

    env = Environment()

    basic0 = ExampleTimedIdentifiable("Hello I'm basic 0!")
    assert basic0.created_at == 0


def test_basic_created_at():

    env = Environment()
    env.step()

    basic1 = ExampleTimedIdentifiable("Hello I'm basic 1!")
    assert basic1.created_at == 1

    for i in range(10):
        env.step()

    basic11 = ExampleTimedIdentifiable("Hello I'm basic 11!")
    assert basic11.created_at == 11
