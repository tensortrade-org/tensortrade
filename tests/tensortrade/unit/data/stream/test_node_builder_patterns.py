
import pytest
import numpy as np
import pandas as pd

from tensortrade.data import DataFeed, Stream, BinOp


def test_lag():

    s1 = Stream([1, 2, 3, 4, 5]).rename("stream")
    assert s1.name == "stream"

    lag = s1.lag()
    assert lag.name == "Lag(stream,1)"

    feed = DataFeed([lag])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["Lag(stream,1)"]]

    assert values == [np.nan, 1, 2, 3, 4]


def test_apply():

    s1 = Stream([1, 4, 9, 16, 25], "s1")

    lag = s1.apply(np.sqrt).rename("apply")

    feed = DataFeed([lag])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["apply"]]

    assert values == [1, 2, 3, 4, 5]


def test_add():

    s1 = Stream([1, 2, 3, 4, 5], "s1")
    assert s1.name == "s1"

    s2 = Stream([1, 2, 3, 4, 5], "s2")
    assert s2.name == "s2"

    s = s1 + s2
    assert s.name == "Add(s1,s2)"

    feed = DataFeed([s])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["Add(s1,s2)"]]

    assert values == [2, 4, 6, 8, 10]


def test_sub():

    s1 = Stream([2, 3, 4, 5, 6], "s1")
    assert s1.name == "s1"

    s2 = Stream([1, 2, 3, 4, 5], "s2")
    assert s2.name == "s2"

    s = s1 - s2
    assert s.name == "Subtract(s1,s2)"

    feed = DataFeed([s])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["Subtract(s1,s2)"]]

    assert values == [1, 1, 1, 1, 1]


def test_truediv():
    s1 = Stream([2, 3, 4, 5, 6], "s1")
    s2 = Stream([2, 3, 4, 5, 6], "s2")

    s = (s1 / s2).rename("div")

    feed = DataFeed([s])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["div"]]

    assert values == [1, 1, 1, 1, 1]

    s1 = Stream([1, 1, 1, 1, 1], "s2")

    s = (5 / s1).rename("div")
    feed = DataFeed([s])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["div"]]

    assert values == [5, 5, 5, 5, 5]

    s1 = Stream([5, 5, 5, 5, 5], "s2")

    s = (s1 / 5).rename("div")
    feed = DataFeed([s])
    feed.compile()

    values = []
    while feed.has_next():
        values += [feed.next()["div"]]

    assert values == [1, 1, 1, 1, 1]


def test_log_returns():

    s1 = Stream([200.23, 198.35, 244.36, 266.30, 250.40], "price")
    assert s1.name == "price"

    lp = s1.log()
    lr = lp - lp.lag()

    feed = DataFeed([lr])
    feed.compile()

    while feed.has_next():
        print(feed.next())

    lr = s1.log().diff().rename("log_return")

    feed = DataFeed([lr])
    feed.compile()

    while feed.has_next():
        print(feed.next())

    # pytest.fail("Failed.")


def test_ewma():

    # adjust: True, ignore_na: True
    v = [5, 2, 4, 6]
    s = Stream(v, "s")

    mean = s.ewm(alpha=0.68, adjust=True, ignore_na=True).mean().rename("mean")

    feed = DataFeed([mean])
    feed.compile()

    expected = list(pd.Series(v).ewm(alpha=0.68, adjust=True, ignore_na=True).mean())

    actual = []
    while feed.has_next():
        actual += [feed.next()["mean"]]

    assert all(np.isclose(actual, expected))

    # adjust: True, ignore_na: False
    v = [5, 2, np.nan, 6]
    s = Stream(v, "s")

    mean = s.ewm(alpha=0.68, adjust=True, ignore_na=False).mean().rename("mean")

    feed = DataFeed([mean])
    feed.compile()

    expected = list(pd.Series(v).ewm(alpha=0.68, adjust=True, ignore_na=False).mean())

    actual = []
    while feed.has_next():
        actual += [feed.next()["mean"]]

    assert all(np.isclose(actual, expected))

    # adjust: True, ignore_na: False
    v = [5, 2, np.nan, 6]
    s = Stream(v, "s")

    mean = s.ewm(alpha=0.68, adjust=False, ignore_na=True).mean().rename("mean")

    feed = DataFeed([mean])
    feed.compile()

    expected = list(pd.Series(v).ewm(alpha=0.68, adjust=False, ignore_na=True).mean())

    actual = []
    while feed.has_next():
        actual += [feed.next()["mean"]]

    assert all(np.isclose(actual, expected))

    # adjust: True, ignore_na: False
    v = [5, 2, np.nan, 6]
    s = Stream(v, "s")

    mean = s.ewm(alpha=0.68, adjust=False, ignore_na=False).mean().rename("mean")

    feed = DataFeed([mean])
    feed.compile()

    expected = list(pd.Series(v).ewm(alpha=0.68, adjust=False, ignore_na=False).mean())

    actual = []
    while feed.has_next():
        actual += [feed.next()["mean"]]

    assert all(np.isclose(actual, expected))


@pytest.mark.skip(reason="This is equal but says nans are not equal.")
def test_ewa_beginning_na():
    # adjust: True, ignore_na: False
    v = [np.nan, 2, np.nan, 6, 8, 5]
    s = Stream(v).rename("s")

    specs = [
        {"alpha": 0.68, "adjust": True, "ignore_na": True, "min_periods": 3},
        {"alpha": 0.68, "adjust": True, "ignore_na": False, "min_periods": 3},
        {"alpha": 0.68, "adjust": False, "ignore_na": True, "min_periods": 3},
        {"alpha": 0.68, "adjust": False, "ignore_na": False, "min_periods": 3}
    ]

    for spec in specs:
        d = spec.copy()
        d["warmup"] = d["min_periods"]
        del d["min_periods"]
        mean = s.ewm(**d).mean().rename("mean")

        feed = DataFeed([mean])
        feed.compile()

        expected = list(pd.Series(v).ewm(**spec).mean())

        actual = []
        while feed.has_next():
            actual += [feed.next()["mean"]]

        assert all(np.isclose(actual, expected))


@pytest.mark.skip(reason="Data module refactor will fix this.")
def test_ewmv_biased():

    # bias: True
    v = [np.nan, 2, np.nan, 6, 8, 5]
    s = Stream(v).rename("s")

    specs = [
        {"alpha": 0.68, "adjust": True, "ignore_na": True, "min_periods": 3},
        {"alpha": 0.68, "adjust": True, "ignore_na": False, "min_periods": 3},
        {"alpha": 0.68, "adjust": False, "ignore_na": True, "min_periods": 3},
        {"alpha": 0.68, "adjust": False, "ignore_na": False, "min_periods": 3}
    ]

    for spec in specs:
        d = spec.copy()
        d["warmup"] = d["min_periods"]
        del d["min_periods"]
        var = s.ewm(**d).var(bias=True).rename("var")

        feed = DataFeed([var])
        feed.compile()

        expected = list(pd.Series(v).ewm(**spec).var(bias=True))

        actual = []
        while feed.has_next():
            actual += [feed.next()["var"]]

        assert all(np.isclose(actual, expected))


@pytest.mark.skip(reason="This is equal but says nans are not equal.")
def test_emwmv_unbiased():

    # bias: True
    v = [np.nan, 2, np.nan, 6, 8, 5]
    s = Stream(v, "s")

    specs = [
        {"alpha": 0.68, "adjust": True, "ignore_na": True, "min_periods": 3},
        {"alpha": 0.68, "adjust": True, "ignore_na": False, "min_periods": 3},
        {"alpha": 0.68, "adjust": False, "ignore_na": True, "min_periods": 3},
        {"alpha": 0.68, "adjust": False, "ignore_na": False, "min_periods": 3}
    ]

    for spec in specs:
        d = spec.copy()
        d["warmup"] = d["min_periods"]
        del d["min_periods"]
        var = s.ewm(**d).var(bias=False).rename("var")

        feed = DataFeed([var])
        feed.compile()

        expected = list(pd.Series(v).ewm(**spec).var(bias=False))

        actual = []
        while feed.has_next():
            actual += [feed.next()["var"]]

        assert all(np.isclose(actual, expected))


def test_min():

    s1 = Stream([1, 2, 3, 4], "s1")
    s2 = Stream([1, 4, 3, 2], "s2")

    m = s1.min(s2).rename("min")

    feed = DataFeed([m])
    feed.compile()

    expected = [1, 2, 3, 2]
    actual = []
    while feed.has_next():
        actual += [feed.next()["min"]]

    assert actual == expected


def test_max():

    s1 = Stream([1, 2, 3, 4], "s1")
    s2 = Stream([1, 4, 3, 2], "s2")

    m = s1.max(s2).rename("max")

    feed = DataFeed([m])
    feed.compile()

    expected = [1, 4, 3, 4]
    actual = []
    while feed.has_next():
        actual += [feed.next()["max"]]

    assert actual == expected


def test_clamp_min():

    s = Stream([1, -2, -3, 0, 5], "s")

    m = s.clamp_min(0).rename("clamp_min")

    feed = DataFeed([m])
    feed.compile()

    expected = [1, 0, 0, 0, 5]
    actual = []
    while feed.has_next():
        actual += [feed.next()["clamp_min"]]

    assert actual == expected


def test_clamp_max():

    s = Stream([1, -2, -3, 0, 5], "s")

    m = s.clamp_max(0).rename("clamp_max")

    feed = DataFeed([m])
    feed.compile()

    expected = [0, -2, -3, 0, 0]
    actual = []
    while feed.has_next():
        actual += [feed.next()["clamp_max"]]

    assert actual == expected


def test_fillna():

    s = Stream([1, np.nan, -3, np.nan, 5], "s")

    m = s.fillna(-1).rename("fill")

    feed = DataFeed([m])
    feed.compile()

    expected = [1, -1, -3, -1, 5]
    actual = []
    while feed.has_next():
        actual += [feed.next()["fill"]]

    assert actual == expected
