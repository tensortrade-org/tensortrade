from sys import exec_prefix
import pytest
import numpy as np
import pandas as pd
import tensortrade.exchanges as exchanges

from gym.spaces import Box

from tensortrade.features import FeatureTransformer
from tensortrade.features.stationarity import FractionalDifference

@pytest.fixture
def exchange():
    return exchanges.get('fbm')


@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df


@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/feature_pipeline_output.csv')
    return df

class TestFractionalDifference():
    def test_incremental_difference(self, data_frame):
        transformer = FractionalDifference(
            difference_order=0.5, inplace=True)

        transformed_frame = transformer.transform(data_frame)
        assert transformed_frame is not None

    def test_incremental_difference_inplace_false(self, data_frame, reference_frame):
        
        frame_tail = data_frame.tail(300)

        transformer = FractionalDifference(
            difference_order=0.5, inplace=False)

        transformed_frame = transformer.transform(frame_tail)
        assert transformed_frame is not None
        assert isinstance(frame_tail, pd.DataFrame)