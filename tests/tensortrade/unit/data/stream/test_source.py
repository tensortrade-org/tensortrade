

from tensortrade.data import Stream


def test_array_init():

    array_ds = Stream('a', [1, 2, 3])

    assert array_ds
    assert array_ds._array == [1, 2, 3]
    assert array_ds._cursor == 0


def test_array_next():

    array_ds = Stream('a', [1, 2, 3])

    next_value = array_ds.forward()

    assert next_value == 1


def test_array_reset():
    array_ds = Stream('a', [1, 2, 3])
    assert array_ds.forward() == 1
    assert array_ds.forward() == 2
    assert array_ds.forward() == 3

    array_ds.reset()
    assert array_ds.forward() == 1
    assert array_ds.forward() == 2
    assert array_ds.forward() == 3
