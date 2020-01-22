
import pandas as pd
import numpy as np


from tensortrade.data import Stream, DataFrameSource


def test_array_init():

    array_ds = Stream('a', [1, 2, 3])

    assert array_ds
    assert array_ds._array == [1, 2, 3]
    assert array_ds._cursor == 0


def test_array_next():

    array_ds = Stream('a', [1, 2, 3])

    next_value = array_ds.next()

    assert next_value == {'a': 1}


def test_array_reset():
    array_ds = Stream('a', [1, 2, 3])
    assert array_ds.next() == {'a': 1}
    assert array_ds.next() == {'a': 2}
    assert array_ds.next() == {'a': 3}

    array_ds.reset()
    assert array_ds.next() == {'a': 1}
    assert array_ds.next() == {'a': 2}
    assert array_ds.next() == {'a': 3}


def test_data_frame_init():

    data = np.array([
        [13863.13, 13889., 12952.5, 13480.01, 11484.01],
        [13480.01, 15275., 13005., 14781.51, 23957.87],
        [14781.51, 15400., 14628., 15098.14, 16584.63],
        [15098.14, 15400., 14230., 15144.99, 17980.39],
        [15144.99, 17178., 14824.05, 16960.01, 20781.65]
    ])
    index = pd.Index(
        ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
        name="date"
    )
    columns = ["open", "high", "low", "close", "volume"]
    data_frame = pd.DataFrame(data, index=index, columns=columns)

    data_frame_ds = DataFrameSource(data_frame)

    assert data_frame_ds


def test_data_frame_next():
    data = np.array([
        [13863.13, 13889., 12952.5, 13480.01, 11484.01],
        [13480.01, 15275., 13005., 14781.51, 23957.87],
        [14781.51, 15400., 14628., 15098.14, 16584.63],
        [15098.14, 15400., 14230., 15144.99, 17980.39],
        [15144.99, 17178., 14824.05, 16960.01, 20781.65]
    ])
    index = pd.Index(
        ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
        name="date"
    )
    columns = ["open", "high", "low", "close", "volume"]
    data_frame = pd.DataFrame(data, index=index, columns=columns)

    data_frame_ds = DataFrameSource(data_frame)

    d1 = data_frame_ds.next()
    assert d1 == {k: v for k, v in zip(columns, data[0, :])}


def test_data_frame_reset():
    data = np.array([
        [13863.13, 13889., 12952.5, 13480.01, 11484.01],
        [13480.01, 15275., 13005., 14781.51, 23957.87],
        [14781.51, 15400., 14628., 15098.14, 16584.63],
        [15098.14, 15400., 14230., 15144.99, 17980.39],
        [15144.99, 17178., 14824.05, 16960.01, 20781.65]
    ])
    index = pd.Index(
        ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
        name="date"
    )
    columns = ["open", "high", "low", "close", "volume"]
    data_frame = pd.DataFrame(data, index=index, columns=columns)

    data_frame_ds = DataFrameSource(data_frame)

    for i in range(5):
        assert data_frame_ds.next() == {k: v for k, v in zip(columns, data[i, :])}

    data_frame_ds.reset()
    for i in range(5):
        assert data_frame_ds.next() == {k: v for k, v in zip(columns, data[i, :])}
