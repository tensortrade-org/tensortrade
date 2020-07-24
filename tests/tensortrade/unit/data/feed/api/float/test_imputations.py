

import numpy as np
import pandas as pd

from tensortrade.feed import Stream

from tests.utils.ops import assert_op


def test_fillna():
    array = [-1, np.nan, -3, 4, np.nan]

    s = Stream.source(array, dtype="float")

    w = s.fillna(-1).rename("w")
    expected = list(pd.Series(array).fillna(-1))

    assert_op([w], expected)


def test_ffill():
    array = [-1, np.nan, -3, 4, np.nan]

    s = Stream.source(array, dtype="float")

    w = s.ffill().rename("w")
    expected = list(pd.Series(array).ffill())

    assert_op([w], expected)
