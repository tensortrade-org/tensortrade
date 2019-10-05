import pytest
import numpy as np
import pandas as pd

from gym.spaces import Box

from tensortrade.features import FeatureTransformer
from tensortrade.features.stationarity import FractionalDifference


@pytest.fixture
def data_frame():
    data_frame = pd.DataFrame([{
        'open': 100,
        'low': 50,
        'high': 250,
        'close': 200,
    },
        {
        'open': 200,
        'low': 150,
        'high': 350,
        'close': 300,
    }])

    return data_frame


class TestFractionalDifference():
    def test_incremental_difference(self, data_frame):
        transformer = FractionalDifference(
            difference_order=0.5, inplace=True, all_column_names=data_frame.columns)

        transformed_frame = transformer.transform(data_frame)

        expected_data_frame = pd.DataFrame([{
            'open': -26.20469322,
            'low': -46.15180724,
            'high': 33.63664884,
            'close': 13.68953482,
        },
            {
            'open': 134.53651465,
            'low': 118.24976426,
            'high': 183.39676584,
            'close': 167.11001545,
        }])

        assert np.allclose(expected_data_frame.values, transformed_frame.values)

        next_frame = pd.DataFrame([{
            'open': 200,
            'low': 150,
            'high': 350,
            'close': 300,
        },
            {
            'open': 300,
            'low': 250,
            'high': 450,
            'close': 400,
        }])

        transformed_frame = transformer.transform(next_frame)

        expected_data_frame = pd.DataFrame([{
            'open': 127.785105,
            'low': 87.031409,
            'high': 250.046192,
            'close': 209.292496,
        },
            {
            'open': 185.484853,
            'low': 166.817514,
            'high': 241.486873,
            'close': 222.819533,
        }])

        assert np.allclose(expected_data_frame.values, transformed_frame.values)

    def test_difference_inplace(self, data_frame):
        transformer = FractionalDifference(
            difference_order=0.5, inplace=True, all_column_names=data_frame.columns)

        transformed_frame = transformer.transform(data_frame)

        expected_data_frame = pd.DataFrame([{
            'open': -26.20469322,
            'low': -46.15180724,
            'high': 33.63664884,
            'close': 13.68953482,
        },
            {
            'open': 134.53651465,
            'low': 118.24976426,
            'high': 183.39676584,
            'close': 167.11001545,
        }])

        assert np.allclose(expected_data_frame.values, transformed_frame.values)

    def test_difference_not_inplace(self, data_frame):
        transformer = FractionalDifference(
            difference_order=0.5, inplace=False, all_column_names=data_frame.columns)

        transformed_frame = transformer.transform(data_frame)

        expected_data_frame = pd.DataFrame([{
            'open': 100,
            'low': 50,
            'high': 250,
            'close': 200,
            'open_diff_0.5': -26.20469322,
            'low_diff_0.5': -46.15180724,
            'high_diff_0.5': 33.63664884,
            'close_diff_0.5': 13.68953482,
        },
            {
            'open': 200,
            'low': 150,
            'high': 350,
            'close': 300,
            'open_diff_0.5': 134.53651465,
            'low_diff_0.5': 118.24976426,
            'high_diff_0.5': 183.39676584,
            'close_diff_0.5': 167.11001545,
        }])

        assert np.allclose(expected_data_frame.values, transformed_frame.values)

    def test_select_correct_columns(self, data_frame):
        transformer = FractionalDifference(
            columns=['open', 'close'], difference_order=0.5, inplace=True, all_column_names=data_frame.columns)

        transformed_frame = transformer.transform(data_frame)

        expected_data_frame = pd.DataFrame([{
            'open': -26.20469322,
            'low': 50,
            'high': 250,
            'close': 13.68953482,
        },
            {
            'open': 134.53651465,
            'low': 150,
            'high': 350,
            'close': 167.11001545,
        }])

        assert np.allclose(expected_data_frame.values, transformed_frame.values)

    def test_transform_space(self, data_frame):
        transformer = FractionalDifference(
            difference_order=0.5, inplace=False, all_column_names=data_frame.columns)

        low = np.array([1E-3, ] * 4 + [1E-3, ])
        high = np.array([1E3, ] * 4 + [1E3, ])

        input_space = Box(low=low, high=high, dtype=np.float16)

        transformed_space = transformer.transform_space(input_space)

        assert transformed_space != input_space
