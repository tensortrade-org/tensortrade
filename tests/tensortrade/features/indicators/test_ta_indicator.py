import pytest
import numpy as np
import pandas as pd
from gym.spaces import Box
from tensortrade.features.indicators.ta_indicator import TAIndicator


@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


class TestTAIndicator:
    indicators_to_test = ['rsi', 'macd', 'ema_indicator']

    def test_ta_indicator(self):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        assert len(test_feature._indicator_names) == 3

    def test_transform(self, data_frame):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        test_feature.transform(data_frame)
        assert set(TestTAIndicator.indicators_to_test).issubset(data_frame.columns)

    def test_transform_single_indicator(self, data_frame):
        test_feature = TAIndicator('rsi')
        test_feature.transform(data_frame)
        assert 'rsi' in data_frame.columns
