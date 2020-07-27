
import numpy as np
import pandas as pd

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


configurations = [
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 0, "adjust": True, "ignore_na": True},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 0, "adjust": True, "ignore_na": False},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 0, "adjust": False, "ignore_na": True},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 0, "adjust": False, "ignore_na": False},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 2, "adjust": True, "ignore_na": True},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 2, "adjust": True, "ignore_na": False},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 2, "adjust": False, "ignore_na": True},
    {"com": None, "span": 3, "halflife": None, "alpha": None, "min_periods": 2, "adjust": False, "ignore_na": False}
]


def test_ewm_mean():

    array = [1, np.nan, 3, 4, 5, 6, np.nan, 7]

    s = Stream.source(array, dtype="float")

    for config in configurations:
        w = s.ewm(**config).mean().rename("w")
        expected = list(pd.Series(array).ewm(**config).mean())

        assert_op([w], expected)


def test_ewm_var():

    array = [1, np.nan, 3, 4, 5, 6, np.nan, 7]

    s = Stream.source(array, dtype="float")

    for config in configurations:
        w = s.ewm(**config).var(bias=False).rename("w")
        expected = list(pd.Series(array).ewm(**config).var(bias=False))

        assert_op([w], expected)

    for config in configurations:
        w = s.ewm(**config).var(bias=True).rename("w")
        expected = list(pd.Series(array).ewm(**config).var(bias=True))

        assert_op([w], expected)


def test_ewm_std():

    array = [1, np.nan, 3, 4, 5, 6, np.nan, 7]

    s = Stream.source(array, dtype="float")

    for config in configurations:
        w = s.ewm(**config).std(bias=False).rename("w")
        expected = list(pd.Series(array).ewm(**config).std(bias=False))

        assert_op([w], expected)

    for config in configurations:
        w = s.ewm(**config).std(bias=True).rename("w")
        expected = list(pd.Series(array).ewm(**config).std(bias=True))

        assert_op([w], expected)
