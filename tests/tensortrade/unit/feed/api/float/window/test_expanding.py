
import numpy as np
import pandas as pd

from itertools import product

from tensortrade.feed import Stream

from tests.utils.ops import assert_op

configurations = [
    {"min_periods": 0},
    {"min_periods": 2}
]

arrays = [
    [1, 2, 3, 4, 5, 6, 7],
    [1, np.nan, 3, 4, 5, 6, np.nan, 7]
]


def test_expanding_count():

    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).mean().rename("w")
        expected = list(pd.Series(array).expanding(**config).mean())

        assert_op([w], expected)


def test_expanding_sum():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).sum().rename("w")
        expected = list(pd.Series(array).expanding(**config).sum())

        assert_op([w], expected)


def test_expanding_mean():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).mean().rename("w")
        expected = list(pd.Series(array).expanding(**config).mean())

        assert_op([w], expected)


def test_expanding_var():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).var().rename("w")
        expected = list(pd.Series(array).expanding(**config).var())

        assert_op([w], expected)


def test_expanding_median():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).median().rename("w")
        expected = list(pd.Series(array).expanding(**config).median())

        assert_op([w], expected)


def test_expanding_std():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).std().rename("w")
        expected = list(pd.Series(array).expanding(**config).std())

        assert_op([w], expected)


def test_expanding_min():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).min().rename("w")
        expected = list(pd.Series(array).expanding(**config).min())

        assert_op([w], expected)


def test_expanding_max():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.expanding(**config).max().rename("w")
        expected = list(pd.Series(array).expanding(**config).max())

        assert_op([w], expected)
