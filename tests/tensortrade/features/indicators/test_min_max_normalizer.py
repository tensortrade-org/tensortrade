import pytest
import ta
import numpy as np
import pandas as pd
from gym.spaces import Box
from tensortrade.features.scalers import MinMaxNormalizer

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df

@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/min_max_transformed.csv')
    return df

class TestMinMaxNormalizer:
    price_columns = ["Open", "High", "Low", "Close", "VolumeFrom", "VolumeTo"]

    def test_min_max_indicator(self, data_frame, reference_frame):
        """  Check that the min-max indicator is to be expected. """
        test_feature = MinMaxNormalizer(columns=TestMinMaxNormalizer.price_columns)
        test_feature.reset()
        
        roundable = 4
        
        
        transformed = test_feature.transform(data_frame)
        assert transformed is not None
        
        transformed = transformed.round(roundable)
        reference_frame = reference_frame.round(roundable)

        close_1 = transformed.Close.values
        close_2 = reference_frame.Close.values

        is_valid = (close_1==close_2).all()
        assert is_valid

    
    def test_min_max_indicator_inplace_false(self, data_frame):
        """ Test that not setting inplace rules equates to more columns after the transformation"""
        test_feature = MinMaxNormalizer(columns=TestMinMaxNormalizer.price_columns, inplace=False)
        test_feature.reset()
        transformed = test_feature.transform(data_frame)
        assert transformed is not None
        assert len(data_frame.columns) != len(transformed.columns)
