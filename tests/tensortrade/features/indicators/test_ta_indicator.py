import pytest
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

import ta

from gym.spaces import Box

from tensortrade.features.indicators.ta_indicator import TAIndicator
#from tensortrade.features.stationarity import FractionalDifference


# @pytest.fixture
# def exchange():
#     return exchanges.get('fbm')
#

@pytest.fixture
def data_frame():
    df = pd.read_csv('../../../../data/input/coinbase-1h-btc-usd.csv')
    return df


class TestTAIndicator():
    indicators_to_test = ["AO", "MFI", "RSI"]

    def test_ta_indicator(self):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        assert len(test_feature._indicator_names) == 3
        assert ('AO', ta.ao, ['High', 'Close']) in test_feature.indicators

    def test_transform_space(self):
        assert 0

    def test_transform(self, data_frame):
        test_feature = TAIndicator(TestTAIndicator.indicators_to_test)
        df = data_frame
        test_feature.transform(df, None)
        assert set(TestTAIndicator.indicators_to_test).issubset(df.columns)





