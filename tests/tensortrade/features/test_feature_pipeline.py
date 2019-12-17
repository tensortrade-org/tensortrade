import pytest
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

from typing import List
from itertools import repeat
from gym.spaces import Space, Box

from tensortrade import TradingContext
from tensortrade.features import FeaturePipeline, FeatureTransformer
from tensortrade.features.stationarity import FractionalDifference


@pytest.fixture
def exchange():
    return exchanges.get('fbm')


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


class TestFeaturePipeline:

    def test_incremental_transform(self, data_frame, exchange):
        difference_all = FractionalDifference(
            difference_order=0.5, inplace=True)

        feature_pipeline = FeaturePipeline(steps=[difference_all])

        transformed_frame = feature_pipeline.transform(data_frame, exchange.generated_space)

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

        transformed_frame = feature_pipeline.transform(next_frame, exchange.generated_space)

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

        import pandas as pd


class Identity(FeatureTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


def test_injects_feature_transformation_with_context():

    config = {
        'features': {
            'shape': (90, 70)
        }
    }

    with TradingContext(**config):

        transformer = Identity()

        assert hasattr(transformer.context, 'shape')
        assert transformer.context.shape == (90, 70)


def test_injects_feature_pipeline_with_context():

    config = {
        'features': {
            'shape': (90, 70)
        }
    }

    with TradingContext(**config):

        steps = list(repeat(Identity(), 5))
        pipeline = FeaturePipeline(steps)
        assert hasattr(pipeline.context, 'shape')
        assert pipeline.context.shape == (90, 70)
