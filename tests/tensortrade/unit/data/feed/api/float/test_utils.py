
import numpy as np
import pandas as pd

from itertools import product

from tensortrade.feed import Stream

from tests.utils.ops import assert_op

arrays = [
    [-1.5, 2.2, -3.3, 4.7, -5.1, 7.45, 8.8],
    [-1.2, 2.3, np.nan, 4.4, -5.5, np.nan, np.nan],
]


def test_ceil():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.ceil().rename("w")
        expected = list(pd.Series(array).apply(np.ceil))

        assert_op([w], expected)


def test_floor():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.floor().rename("w")
        expected = list(pd.Series(array).apply(np.floor))

        assert_op([w], expected)


def test_sqrt():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.sqrt().rename("w")
        expected = list(pd.Series(array).apply(np.sqrt))

        assert_op([w], expected)


def test_square():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.square().rename("w")
        expected = list(pd.Series(array).apply(np.square))

        assert_op([w], expected)


def test_log():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.log().rename("w")
        expected = list(pd.Series(array).apply(np.log))

        assert_op([w], expected)


def test_pct_change():
    configs = [
        {"periods": 1, "fill_method": None},
        {"periods": 1, "fill_method": "pad"},
        {"periods": 1, "fill_method": "ffill"},
        {"periods": 2, "fill_method": None},
        {"periods": 2, "fill_method": "pad"},
        {"periods": 2, "fill_method": "ffill"},
    ]

    for array, config in product(arrays, configs):

        s = Stream.source(array, dtype="float")
        w = s.pct_change(**config).rename("w")
        expected = list(pd.Series(array).pct_change(**config))

        print(config)
        assert_op([w], expected)


def test_diff():

    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.diff(periods=1).rename("w")
        expected = list(pd.Series(array).diff(periods=1))

        assert_op([w], expected)

    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.diff(periods=2).rename("w")
        expected = list(pd.Series(array).diff(periods=2))

        assert_op([w], expected)
