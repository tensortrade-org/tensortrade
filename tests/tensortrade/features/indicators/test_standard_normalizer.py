import pytest
import ta
import numpy as np
import pandas as pd
from tensortrade.features.scalers import StandardNormalizer

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df

@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/standard_transformed.csv')
    return df


class TestStandardNormalizer:
    price_columns = ["Open", "High", "Low", "Close"]
    indicators = ["EMA", "RSI"]


    def test_standard_normalizer(self, data_frame, reference_frame):
        """ Here we check to """
        standard = StandardNormalizer(columns=TestStandardNormalizer.price_columns)
        transformed = standard.transform(data_frame)
        assert transformed is not None
        assert len(data_frame) == len(transformed)
        assert len(data_frame.columns) == len(transformed.columns)

        # Now compare the columns
    

    def test_standard_normalizer_inplace(self, data_frame):
        """ Run a standard scaler transform and expect it to fail with a NotImplementedError"""
        standard = StandardNormalizer(columns=TestStandardNormalizer.price_columns, inplace=False)
        transformed = standard.transform(data_frame)
        assert transformed is not None


        # TODO: Create a dataframe and save it in storage with

        assert len(data_frame) == len(transformed)
        assert len(data_frame.columns) != len(transformed.columns)