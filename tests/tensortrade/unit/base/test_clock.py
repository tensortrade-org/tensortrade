

import tensortrade.core.clock as clock

from tensortrade.core import Clock, TimeIndexed


def test_basic_clock_init():
    clock = Clock()

    assert clock
    assert clock.start == 0
    assert clock.step == 0


def test_basic_clock_increment():
    clock = Clock()

    clock.increment()

    assert clock.step == 1


def test_time_indexed_init():

    class FirstExample(TimeIndexed):

        def __init__(self, msg):
            self.msg = msg

    class SecondExample(TimeIndexed):

        def __init__(self, msg):
            self.msg = msg

    example1 = FirstExample("Example 1")
    example2 = SecondExample("Example 2")

    assert example1.clock
    assert example1.clock.start == 0
    assert example1.clock.step == 0

    assert example2.clock
    assert example2.clock.start == 0
    assert example2.clock.step == 0

    assert example1.clock == example2.clock


def test_time_indexed_increment():

    class FirstExample(TimeIndexed):

        def __init__(self, msg):
            self.msg = msg

    class SecondExample(TimeIndexed):

        def __init__(self, msg):
            self.msg = msg

    example1 = FirstExample("Example 1")
    example2 = SecondExample("Example 2")

    example1.clock.increment()

    assert example1.clock.start == 0
    assert example2.clock.start == 0
    assert example1.clock.step == 1
    assert example2.clock.step == 1

    example1.clock.reset()
