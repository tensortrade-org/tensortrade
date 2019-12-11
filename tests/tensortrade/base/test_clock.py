

import tensortrade.base.clock as clock

from tensortrade.base.clock import BasicClock, TimeIndexed


def test_basic_clock_init():
    clock = BasicClock()

    assert clock
    assert clock.start == 0
    assert clock.now() == 0


def test_basic_clock_increment():
    clock = BasicClock()

    clock.increment()

    assert clock.now() == 1


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
    assert example1.clock.now() == 0

    assert example2.clock
    assert example2.clock.start == 0
    assert example2.clock.now() == 0

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
    assert example1.clock.now() == 1
    assert example2.clock.now() == 1

    example1.clock.reset()


def test_time_indexed_increment_from_module():

    class FirstExample(TimeIndexed):

        def __init__(self, msg):
            self.msg = msg

    class SecondExample(TimeIndexed):

        def __init__(self, msg):
            self.msg = msg

    print(TimeIndexed.clock.now())
    example1 = FirstExample("Example 1")
    example2 = SecondExample("Example 2")
    print(example1.clock.now())

    clock.increment()

    assert example1.clock.start == 0
    assert example2.clock.start == 0
    assert example1.clock.now() == 1
    assert example2.clock.now() == 1

    clock.reset()

    assert example1.clock.start == 0
    assert example2.clock.start == 0
    assert example1.clock.now() == 0
    assert example2.clock.now() == 0