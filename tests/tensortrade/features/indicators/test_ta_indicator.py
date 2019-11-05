import pytest
import numpy as np
import pandas as pd
from gym.spaces import Box
from tensortrade.features.indicators.ta_indicator import TAIndicator


@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


@pytest.fixture
def input_space():
    return Box(low=0.0, high=100, shape=(5, 5), dtype=np.float32)


class TestTAIndicator:
    indicators_to_test = ['rsi', 'macd', 'ema_indicator']

    def test_ta_indicator(self):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        assert len(test_feature._indicator_names) == 3

    def test_transform_space(self, input_space, data_frame):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        column_names = data_frame.columns
        output = test_feature.transform_space(input_space, column_names)
        assert output.shape[0] > input_space.shape[0]
        assert output.shape[1] == input_space.shape[1]

    def test_transform(self, data_frame):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        test_feature.transform(data_frame, None)
        assert set(TestTAIndicator.indicators_to_test).issubset(data_frame.columns)

    def test_transform_single_indicator(self, data_frame):
        test_feature = TAIndicator('rsi')
        test_feature.transform(data_frame, None)
        assert 'rsi' in data_frame.columns

    # def test_add_volatility_ta(self, data_frame):
    #     test_feature = TAIndicator('add_volatility_ta')
    #     print(test_feature._indicator_names)
    #     test_feature.transform(data_frame, None)
    #     print(data_frame.columns)
    #     print(data_frame.dtypes)
