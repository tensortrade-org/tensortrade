import pytest
import pandas as pd
import numpy as np
from tensortrade.features.indicators import TAlibIndicator
# from pandas.util.testing import assert_frame_equal

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df

@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/ta_transformed.csv')
    return df


class TestTALibIndicator:
    price_columns = ["open", "high", "low", "close"]
    indicators = ["BBAND", "RSI", "EMA", "SMA", "", None]


    def test_ta_lib_indicator(self, data_frame, reference_frame):
        test_feature = TAlibIndicator(indicators=TestTALibIndicator.indicators)
        transformed = test_feature.transform(data_frame)
        transformed = transformed.dropna()
        reference_frame = reference_frame.drop(reference_frame.columns[0], axis=1)
        
        transformed = transformed.round(3)
        reference_frame = reference_frame.round(3)

        assert len(test_feature._indicator_names) == 4
        assert list(transformed.columns) == list(reference_frame.columns)
        bb_middle1 = np.array(transformed.bb_middle.values, dtype=np.double)
        bb_middle2 = np.array(reference_frame.bb_middle.values, dtype=np.double)
        middle_diff = (bb_middle1==bb_middle2).all()
        assert middle_diff



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