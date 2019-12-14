import pytest
import ta
import numpy as np
import pandas as pd
from tensortrade.features.scalers import StandardNormalizer, ComparisonNormalizer

@pytest.fixture
def data_frame():
    df = pd.read_csv('tests/data/input/coinbase-1h-btc-usd.csv')
    return df

@pytest.fixture
def reference_frame():
    df = pd.read_csv('tests/data/outputs/com_norm_transformed.csv')
    return df


class TestComparisonNormalizer:
    price_columns = ["Open", "High", "Low", "Close"]

    def test_comparison_normalizer(self, data_frame, reference_frame):
        """ Here we check to """
        comparison = ComparisonNormalizer(columns=TestComparisonNormalizer.price_columns, comparison_column="Close")
        transformed = comparison.transform(data_frame)

        transformed = transformed.round(4)
        reference_frame = reference_frame.round(4)

        assert transformed is not None
        assert len(data_frame) == len(transformed)
        assert len(data_frame.columns) == len(transformed.columns)
        assert len(transformed) == len(reference_frame)

        close_1 = transformed.Close.values
        close_2 = reference_frame.Close.values

        comparison_equal = (close_1==close_2).all()
        assert comparison_equal

    def test_comparison_normalizer_inplace(self, data_frame):
        """ Run a standard scaler transform and expect it to fail with a NotImplementedError"""
        comparison = ComparisonNormalizer(columns=TestComparisonNormalizer.price_columns, comparison_column="Close", inplace=False)
        transformed = comparison.transform(data_frame)
        assert transformed is not None
        # TODO: Create a dataframe and save it in storage with
        assert len(data_frame) == len(transformed)
        assert len(data_frame.columns) != len(transformed.columns)