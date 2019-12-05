import os
import pytest
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

from gym.spaces import Box

from tensortrade.features import FeaturePipeline, FeatureTransformer
from tensortrade.features.stationarity import FractionalDifference

path = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def exchange():
    return exchanges.get('fbm')


@pytest.fixture
def data_frame():
    data_frame = pd.read_csv('../../data/input/coinbase-1h-btc-usd.csv', skiprows=1)
    return data_frame



class TestFeaturePipeline:


    def test_incremental_transform(self, data_frame, exchange):
        exchange.reset()
        difference_all = FractionalDifference(
            difference_order=0.5, inplace=True)

        feature_pipeline = FeaturePipeline(steps=[difference_all])


        obs = exchange._next_observation()
        transformed_frame = feature_pipeline.transform(data_frame)

        assert transformed_frame

        # assert np.allclose(expected_data_frame.values, transformed_frame.values)