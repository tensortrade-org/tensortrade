import pytest
import os
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

import ta

from gym.spaces import Box

from tensortrade.features.indicators.ta_indicator import TAIndicator


@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


@pytest.fixture
def input_space():
    return Box(low=0.0, high=100, shape=(5,5), dtype=np.float32)


class TestTAIndicator:
    indicators_to_test = ["AO", "MFI", "RSI"]

    def test_ta_indicator(self):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        assert len(test_feature._indicator_names) == 3
        assert ('AO', ta.ao, ['High', 'Close']) in test_feature.indicators

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





