import os
import pytest
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

from gym.spaces import Box

from tensortrade.features import FeaturePipeline
from tensortrade.features.indicators import TAlibIndicator
from tensortrade.features.scalers import MinMaxNormalizer

path = os.path.dirname(os.path.abspath(__file__))



@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


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

        columns = reference_frame.columns
        col_1 = columns[0]
        
        reference_frame = reference_frame.drop(columns=[col_1])
        transformed_frame = feature_pipeline.transform(data_frame)
        
        transformed_frame = transformed_frame.dropna()
        reference_frame = reference_frame.dropna()
        # We round to 4 significant digits
        significance = 10
        transformed_frame = transformed_frame.round(significance)
        reference_frame = reference_frame.round(significance)

        bb_middle1 = reference_frame.bb_middle.values
        bb_middle2 = transformed_frame.bb_middle.values
        
        
        is_valid = (bb_middle1==bb_middle2).all()
        assert is_valid