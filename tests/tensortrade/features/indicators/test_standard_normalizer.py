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


    def test_standard_normalizer(self, data_frame):
        standard = StandardNormalizer(TestStandardNormalizer.price_columns)
        # standard.transform(data_frame)
        # assert True == False

    # @pytest.mark.xfail
    # def test_transform(self, data_frame):
    #     test_feature = TAlibIndicator(TestTAIndicator.indicators)
    #     test_feature.transform(data_frame)
    #     assert set(TestTAIndicator.indicators).issubset(data_frame.columns)




""" 
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import TAlibIndicator
price_columns = ["open", "high", "low", "close"]
normalize_price = MinMaxNormalizer(price_columns)
moving_averages = TAlibIndicator(["EMA", "RSI", "BB"])
difference_all = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price,
                                          moving_averages,
                                          difference_all])
exchange.feature_pipeline = feature_pipeline
"""