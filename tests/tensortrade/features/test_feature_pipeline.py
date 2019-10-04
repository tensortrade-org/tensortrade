import pytest
import numpy as np
import pandas as pd

from gym.spaces import Box

from tensortrade.features import FeaturePipeline, FeatureTransformer
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


class TestFeaturePipeline():
    def test_incremental_transform(self, data_frame):
        difference_all = FractionalDifference(
            difference_order=0.5, inplace=True, all_column_names=data_frame.columns)

        feature_pipeline = FeaturePipeline(steps=[difference_all])

        transformed_frame = feature_pipeline.transform(data_frame)

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

        transformed_frame = feature_pipeline.transform(next_frame)

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

    def test_transform_space(self, data_frame):
        difference_all = FractionalDifference(
            difference_order=0.5, inplace=False, all_column_names=data_frame.columns)

        feature_pipeline = FeaturePipeline(steps=[difference_all])

        low = np.array([1E-3, ] * 4 + [1E-3, ])
        high = np.array([1E3, ] * 4 + [1E3, ])

        input_space = Box(low=low, high=high, dtype=np.float16)

        transformed_space = feature_pipeline.transform_space(input_space)

        assert transformed_space != input_space
