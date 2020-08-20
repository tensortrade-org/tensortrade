
from itertools import product

import pytest
import numpy as np
import pandas as pd

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


arrays = [
    [1, 2, 3, 4, 5, 6, 7],
    [1, np.nan, 3, 4, 5, 6, np.nan, 7]
]

configurations = [
    {"window": 2, "min_periods": 0},
    {"window": 2, "min_periods": 1},
    {"window": 2, "min_periods": 2},
    {"window": 3, "min_periods": 0},
    {"window": 3, "min_periods": 1},
    {"window": 3, "min_periods": 2},
    {"window": 3, "min_periods": 3},
    {"window": 4, "min_periods": 0},
    {"window": 4, "min_periods": 1},
    {"window": 4, "min_periods": 2},
    {"window": 4, "min_periods": 3},
    {"window": 4, "min_periods": 4},
]


@pytest.mark.skip("Details don't completely match. Needs to be figured out.")
def test_rolling_count():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).count().rename("w")
        expected = list(pd.Series(array).rolling(**config).count())

        assert_op([w], expected)


def test_rolling_sum():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).sum().rename("w")
        expected = list(pd.Series(array).rolling(**config).sum())

        assert_op([w], expected)


def test_rolling_mean():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).mean().rename("w")
        expected = list(pd.Series(array).rolling(**config).mean())

        assert_op([w], expected)


def test_rolling_var():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).var().rename("w")
        expected = list(pd.Series(array).rolling(**config).var())

        assert_op([w], expected)


def test_rolling_median():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).median().rename("w")
        expected = list(pd.Series(array).rolling(**config).median())

        assert_op([w], expected)


def test_rolling_std():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).std().rename("w")
        expected = list(pd.Series(array).rolling(**config).std())

        assert_op([w], expected)


def test_rolling_min():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).min().rename("w")
        expected = list(pd.Series(array).rolling(**config).min())

        assert_op([w], expected)


def test_rolling_max():
    for array, config in product(arrays, configurations):
        s = Stream.source(array, dtype="float")
        w = s.rolling(**config).max().rename("w")
        expected = list(pd.Series(array).rolling(**config).max())

        assert_op([w], expected)
