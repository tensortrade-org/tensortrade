import pytest
import numpy as np
from tensortrade.env.generic.utils.running_stats import Welfords, PctChange


class TestRunningStats:
    def test_simple(self):
        data_series = np.array([1,2,3,4,5,6,7,8,9])
        welfords = Welfords()

        mean = np.mean(data_series)
        variance = np.var(data_series)
        std = np.std(data_series)

        for i in data_series:
            welfords.include(i)

        assert mean == welfords.mean
        assert variance == welfords.variance
        assert std == welfords.std

    def test_windowed_welfords(self):
        window = 3
        window_data = []
        data_series = np.array([1,2,3,4,5,6,7,8,9])
        welfords = Welfords(window_size=window)

        for i in data_series:
            welfords.include(i)
            window_data.append(i)
            if len(window_data) > window:
                del window_data[0]

        assert np.mean(window_data) == welfords.mean
        assert np.var(window_data) == welfords.variance
        assert np.std(window_data) == welfords.std
