
import numpy as np
import pandas as pd

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


arrays = [
    [-1, 2, -3, 4, -5, 7, 8],
    [-1, 2, np.nan, 4, -5, np.nan, np.nan],
]


def test_cumsum():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.cumsum().rename("w")
        expected = list(pd.Series(array).cumsum())

        assert_op([w], expected)


def test_cumprod():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.cumprod().rename("w")
        expected = list(pd.Series(array).cumprod())

        assert_op([w], expected)


def test_cummin():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.cummin(skipna=True).rename("w")
        expected = list(pd.Series(array).cummin(skipna=True))

        assert_op([w], expected)

    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.cummin(skipna=False).rename("w")
        expected = list(pd.Series(array).cummin(skipna=False))

        assert_op([w], expected)


def test_cummax():
    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.cummax(skipna=True).rename("w")
        expected = list(pd.Series(array).cummax(skipna=True))

        assert_op([w], expected)

    for array in arrays:
        s = Stream.source(array, dtype="float")
        w = s.cummin(skipna=False).rename("w")
        expected = list(pd.Series(array).cummin(skipna=False))

        assert_op([w], expected)
