import pytest
import ta
import numpy as np
import pandas as pd
from tensortrade.features.scalers import StandardNormalizer, PercentChangeNormalizer

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df

@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/pct_transforms.csv')
    return df


class TestPercentNormalizer:
    price_columns = ["Open", "High", "Low", "Close"]
    indicators = ["EMA", "RSI"]


    def test_percent_normalizer(self, data_frame, reference_frame):
        """ Here we check to """
        pct_change_norm = PercentChangeNormalizer()
        transformed = pct_change_norm.transform(data_frame)
        assert transformed is not None
        assert len(data_frame) == len(transformed)
        assert len(data_frame.columns) == len(transformed.columns)



        # This needs to be rounded due to csv importing bug.
        reference_frame = reference_frame.round(5)
        transformed = transformed.round(5)

        close1 = reference_frame.Close.values
        close2 = transformed.Close.values
        is_valid = (close1==close2).all()
        assert is_valid
    

    def test_percent_normalizer_inplace(self, data_frame):
        """ Run a standard scaler transform and expect it to fail with a NotImplementedError"""
        pct_change_norm = PercentChangeNormalizer(inplace=False)
        transformed = pct_change_norm.transform(data_frame)
        assert transformed is not None


        # TODO: Create a dataframe and save it in storage with

        assert len(data_frame) == len(transformed)
        assert len(data_frame.columns) != len(transformed.columns)