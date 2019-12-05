import pytest
import pandas as pd
from tensortrade.features.indicators import TAlibIndicator

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


class TestTAIndicator:
    price_columns = ["open", "high", "low", "close"]
    indicators = ["EMA", "RSI", "BBANDS"]


    def test_ta_indicator(self, data_frame):
        test_feature = TAlibIndicator(indicators=TestTAIndicator.indicators)
        test_feature.transform(data_frame)
        assert len(test_feature._indicator_names) == 3



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