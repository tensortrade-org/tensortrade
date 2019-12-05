import pytest
import ta
import numpy as np
import pandas as pd
from tensortrade.features.scalers import StandardNormalizer

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


class TestStandardNormalizer:
    price_columns = ["open", "high", "low", "close"]
    indicators = ["EMA", "RSI"]


    @pytest.mark.xfail(raises=NotImplementedError)
    def test_standard_normalizer(self, data_frame):
        """ Run a standard scaler transform and expect it to fail with a NotImplementedError"""
        standard = StandardNormalizer(TestStandardNormalizer.price_columns)
        transformed = standard.transform(data_frame)
        assert transformed is not None