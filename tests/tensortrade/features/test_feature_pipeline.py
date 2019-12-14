import os
import pytest
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

from gym.spaces import Box

from tensortrade.features import FeaturePipeline
from tensortrade.features.indicators import TAlibIndicator
from tensortrade.features.scalers import MinMaxNormalizer
# from tensortrade.features.stationarity import FractionalDifference

path = os.path.dirname(os.path.abspath(__file__))



@pytest.fixture
def data_frame():
    data_frame = pd.read_csv('../../data/input/coinbase-1h-btc-usd.csv', skiprows=1)
    return data_frame


@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/feature_pipeline_output.csv')
    return df


class TestFeaturePipeline:


    def test_full_ta_min_max_pipeline(self, data_frame, reference_frame):
        ta_indicator = TAlibIndicator(indicators=["BBAND", "RSI", "EMA", "SMA"])
        min_max = MinMaxNormalizer()
        feature_pipeline = FeaturePipeline([
            ta_indicator, 
            min_max
        ])
        transformed_frame = feature_pipeline.transform(data_frame)
        dropped_transformed = transformed_frame.dropna()
        # We round to 4 significant digits
        dropped_transformed = dropped_transformed.round(4)
        dropped_transformed = dropped_transformed.reset_index()
        
        reference_frame = reference_frame.round(4)
        reference_frame = reference_frame.reset_index()

        assert reference_frame.equals(dropped_transformed)