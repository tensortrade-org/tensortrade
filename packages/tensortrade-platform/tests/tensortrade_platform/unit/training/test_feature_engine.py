"""Tests for FeatureEngine."""

import numpy as np
import pandas as pd
import pytest

from tensortrade.training.feature_engine import FEATURE_CATALOG, FeatureEngine


@pytest.fixture
def engine():
    return FeatureEngine()


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.3),
            "low": close - abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.uniform(100, 1000, n),
        }
    )


class TestListAvailable:
    def test_list_available_returns_catalog(self, engine):
        catalog = engine.list_available()
        assert len(catalog) > 0
        assert catalog == FEATURE_CATALOG

    def test_list_available_contains_expected_types(self, engine):
        catalog = engine.list_available()
        types = {entry["type"] for entry in catalog}
        expected = {
            "returns",
            "rsi",
            "sma_trend",
            "trend_strength",
            "volatility",
            "volume_ratio",
            "bollinger_position",
            "macd",
            "atr",
            "stochastic",
            "obv",
            "roc",
            "cci",
        }
        assert expected == types

    def test_catalog_entries_have_required_keys(self, engine):
        for entry in engine.list_available():
            assert "type" in entry
            assert "name" in entry
            assert "description" in entry
            assert "params" in entry


class TestComputeReturns:
    def test_compute_returns(self, engine, sample_df):
        specs = [{"type": "returns", "periods": [1, 4, 12], "normalize": "tanh"}]
        result = engine.compute(sample_df, specs)
        assert "ret_1h" in result.columns
        assert "ret_4h" in result.columns
        assert "ret_12h" in result.columns
        # tanh output is bounded in [-1, 1]
        assert result["ret_1h"].dropna().between(-1, 1).all()

    def test_compute_returns_no_normalize(self, engine, sample_df):
        specs = [{"type": "returns", "periods": [1], "normalize": "none"}]
        result = engine.compute(sample_df, specs)
        assert "ret_1h" in result.columns


class TestComputeRSI:
    def test_compute_rsi_normalized(self, engine, sample_df):
        specs = [{"type": "rsi", "period": 14, "normalize": True}]
        result = engine.compute(sample_df, specs)
        assert "rsi" in result.columns
        rsi_vals = result["rsi"].dropna()
        assert rsi_vals.between(-1, 1).all()

    def test_compute_rsi_unnormalized(self, engine, sample_df):
        specs = [{"type": "rsi", "period": 14, "normalize": False}]
        result = engine.compute(sample_df, specs)
        assert "rsi" in result.columns
        rsi_vals = result["rsi"].dropna()
        # Raw RSI is in [0, 100]
        assert rsi_vals.between(0, 100).all()


class TestComputeSMATrend:
    def test_compute_sma_trend(self, engine, sample_df):
        specs = [{"type": "sma_trend", "fast": 10, "slow": 50, "normalize": "tanh"}]
        result = engine.compute(sample_df, specs)
        assert "trend_10" in result.columns
        assert "trend_50" in result.columns
        assert result["trend_10"].dropna().between(-1, 1).all()
        assert result["trend_50"].dropna().between(-1, 1).all()


class TestComputeTrendStrength:
    def test_compute_trend_strength(self, engine, sample_df):
        specs = [{"type": "trend_strength", "fast": 20, "slow": 50}]
        result = engine.compute(sample_df, specs)
        assert "trend_strength" in result.columns
        # tanh bounded
        assert result["trend_strength"].dropna().between(-1, 1).all()


class TestComputeVolatility:
    def test_compute_volatility(self, engine, sample_df):
        specs = [{"type": "volatility", "period": 24, "rolling_norm_period": 72}]
        result = engine.compute(sample_df, specs)
        assert "vol" in result.columns
        assert "vol_norm" in result.columns

    def test_compute_volatility_no_normalization(self, engine, sample_df):
        specs = [{"type": "volatility", "period": 24, "rolling_norm_period": 0}]
        result = engine.compute(sample_df, specs)
        assert "vol" in result.columns
        assert "vol_norm" not in result.columns


class TestComputeVolumeRatio:
    def test_compute_volume_ratio(self, engine, sample_df):
        specs = [{"type": "volume_ratio", "period": 20}]
        result = engine.compute(sample_df, specs)
        assert "vol_ratio" in result.columns

    def test_compute_volume_ratio_no_volume_col(self, engine):
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        specs = [{"type": "volume_ratio", "period": 3}]
        result = engine.compute(df, specs)
        assert "vol_ratio" in result.columns
        assert (result["vol_ratio"] == 0.0).all()


class TestComputeBollingerPosition:
    def test_compute_bollinger_position(self, engine, sample_df):
        specs = [{"type": "bollinger_position", "period": 20, "std_dev": 2.0}]
        result = engine.compute(sample_df, specs)
        assert "bb_pos" in result.columns
        bb_vals = result["bb_pos"].dropna()
        assert bb_vals.between(0, 1).all()


class TestComputeMACD:
    def test_compute_macd_columns(self, engine, sample_df):
        specs = [{"type": "macd", "fast": 12, "slow": 26, "signal": 9}]
        result = engine.compute(sample_df, specs)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_compute_macd_bounded(self, engine, sample_df):
        specs = [{"type": "macd"}]
        result = engine.compute(sample_df, specs)
        for col in ["macd_line", "macd_signal", "macd_hist"]:
            assert result[col].between(-1, 1).all()

    def test_compute_macd_no_nans(self, engine, sample_df):
        specs = [{"type": "macd"}]
        result = engine.compute(sample_df, specs)
        assert result["macd_line"].isna().sum() == 0


class TestComputeATR:
    def test_compute_atr_columns(self, engine, sample_df):
        specs = [{"type": "atr", "period": 14, "rolling_norm_period": 72}]
        result = engine.compute(sample_df, specs)
        assert "atr" in result.columns

    def test_compute_atr_bounded(self, engine, sample_df):
        specs = [{"type": "atr"}]
        result = engine.compute(sample_df, specs)
        assert result["atr"].between(-1, 1).all()

    def test_compute_atr_no_nans(self, engine, sample_df):
        specs = [{"type": "atr"}]
        result = engine.compute(sample_df, specs)
        assert result["atr"].isna().sum() == 0

    def test_compute_atr_fallback_no_high_low(self, engine):
        df = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})
        specs = [{"type": "atr", "period": 14, "rolling_norm_period": 72}]
        result = engine.compute(df, specs)
        assert "atr" in result.columns
        assert result["atr"].isna().sum() == 0


class TestComputeStochastic:
    def test_compute_stochastic_columns(self, engine, sample_df):
        specs = [{"type": "stochastic", "k_period": 14, "d_period": 3}]
        result = engine.compute(sample_df, specs)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_compute_stochastic_bounded(self, engine, sample_df):
        specs = [{"type": "stochastic"}]
        result = engine.compute(sample_df, specs)
        assert result["stoch_k"].between(-1, 1).all()
        assert result["stoch_d"].between(-1, 1).all()

    def test_compute_stochastic_no_nans(self, engine, sample_df):
        specs = [{"type": "stochastic"}]
        result = engine.compute(sample_df, specs)
        assert result["stoch_k"].isna().sum() == 0
        assert result["stoch_d"].isna().sum() == 0

    def test_compute_stochastic_fallback_no_high_low(self, engine):
        df = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})
        specs = [{"type": "stochastic"}]
        result = engine.compute(df, specs)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns


class TestComputeOBV:
    def test_compute_obv_columns(self, engine, sample_df):
        specs = [{"type": "obv", "rolling_norm_period": 20}]
        result = engine.compute(sample_df, specs)
        assert "obv" in result.columns

    def test_compute_obv_bounded(self, engine, sample_df):
        specs = [{"type": "obv"}]
        result = engine.compute(sample_df, specs)
        assert result["obv"].between(-1, 1).all()

    def test_compute_obv_no_nans(self, engine, sample_df):
        specs = [{"type": "obv"}]
        result = engine.compute(sample_df, specs)
        assert result["obv"].isna().sum() == 0

    def test_compute_obv_no_volume_col(self, engine):
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        specs = [{"type": "obv"}]
        result = engine.compute(df, specs)
        assert "obv" in result.columns
        assert (result["obv"] == 0.0).all()


class TestComputeROC:
    def test_compute_roc_columns(self, engine, sample_df):
        specs = [{"type": "roc", "period": 12, "normalize": "tanh"}]
        result = engine.compute(sample_df, specs)
        assert "roc" in result.columns

    def test_compute_roc_bounded(self, engine, sample_df):
        specs = [{"type": "roc"}]
        result = engine.compute(sample_df, specs)
        assert result["roc"].between(-1, 1).all()

    def test_compute_roc_no_normalize(self, engine, sample_df):
        specs = [{"type": "roc", "period": 12, "normalize": "none"}]
        result = engine.compute(sample_df, specs)
        assert "roc" in result.columns

    def test_compute_roc_no_nans(self, engine, sample_df):
        specs = [{"type": "roc"}]
        result = engine.compute(sample_df, specs)
        assert result["roc"].isna().sum() == 0


class TestComputeCCI:
    def test_compute_cci_columns(self, engine, sample_df):
        specs = [{"type": "cci", "period": 20}]
        result = engine.compute(sample_df, specs)
        assert "cci" in result.columns

    def test_compute_cci_bounded(self, engine, sample_df):
        specs = [{"type": "cci"}]
        result = engine.compute(sample_df, specs)
        assert result["cci"].between(-1, 1).all()

    def test_compute_cci_no_nans(self, engine, sample_df):
        specs = [{"type": "cci"}]
        result = engine.compute(sample_df, specs)
        assert result["cci"].isna().sum() == 0

    def test_compute_cci_fallback_no_high_low(self, engine):
        df = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(100) * 0.5)})
        specs = [{"type": "cci"}]
        result = engine.compute(df, specs)
        assert "cci" in result.columns
        assert result["cci"].isna().sum() == 0


class TestComputeMultipleFeatures:
    def test_compute_multiple_features(self, engine, sample_df):
        specs = [
            {"type": "returns", "periods": [1, 4], "normalize": "tanh"},
            {"type": "rsi", "period": 14, "normalize": True},
            {"type": "sma_trend", "fast": 10, "slow": 50, "normalize": "tanh"},
            {"type": "volatility", "period": 24, "rolling_norm_period": 72},
            {"type": "volume_ratio", "period": 20},
            {"type": "bollinger_position", "period": 20, "std_dev": 2.0},
        ]
        result = engine.compute(sample_df, specs)

        expected_cols = ["ret_1h", "ret_4h", "rsi", "trend_10", "trend_50", "vol", "vol_norm", "vol_ratio", "bb_pos"]
        for col in expected_cols:
            assert col in result.columns

    def test_compute_fills_nans(self, engine, sample_df):
        specs = [{"type": "rsi", "period": 14}]
        result = engine.compute(sample_df, specs)
        assert result["rsi"].isna().sum() == 0


class TestPreview:
    def test_preview_returns_dict(self, engine, sample_df):
        specs = [{"type": "rsi", "period": 14}]
        preview = engine.preview(sample_df, specs)
        assert isinstance(preview, dict)
        assert "rows" in preview
        assert "feature_columns" in preview
        assert "stats" in preview
        assert "sample" in preview

    def test_preview_feature_columns(self, engine, sample_df):
        specs = [
            {"type": "rsi", "period": 14},
            {"type": "returns", "periods": [1]},
        ]
        preview = engine.preview(sample_df, specs)
        assert "rsi" in preview["feature_columns"]
        assert "ret_1h" in preview["feature_columns"]

    def test_preview_stats_have_expected_keys(self, engine, sample_df):
        specs = [{"type": "rsi", "period": 14}]
        preview = engine.preview(sample_df, specs)
        assert "rsi" in preview["stats"]
        rsi_stats = preview["stats"]["rsi"]
        assert "mean" in rsi_stats
        assert "std" in rsi_stats
        assert "min" in rsi_stats
        assert "max" in rsi_stats

    def test_preview_sample_rows_limited(self, engine, sample_df):
        specs = [{"type": "rsi", "period": 14}]
        preview = engine.preview(sample_df, specs, sample_rows=10)
        assert len(preview["sample"]) == 10


class TestEmptySpecs:
    def test_empty_specs_returns_df_unchanged(self, engine, sample_df):
        result = engine.compute(sample_df, [])
        # Should have same columns as original (after bfill/ffill)
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)


class TestUnknownFeatureType:
    def test_unknown_feature_type_skipped(self, engine, sample_df):
        specs = [{"type": "nonexistent_feature", "period": 10}]
        result = engine.compute(sample_df, specs)
        # Should return df unchanged (no new columns)
        assert list(result.columns) == list(sample_df.columns)


class TestGetFeatureColumns:
    def test_get_feature_columns(self, engine):
        specs = [
            {"type": "returns", "periods": [1, 4]},
            {"type": "rsi", "period": 14},
            {"type": "sma_trend", "fast": 10, "slow": 50},
            {"type": "trend_strength", "fast": 20, "slow": 50},
            {"type": "volatility", "period": 24, "rolling_norm_period": 72},
            {"type": "volume_ratio", "period": 20},
            {"type": "bollinger_position", "period": 20},
            {"type": "macd"},
            {"type": "atr"},
            {"type": "stochastic"},
            {"type": "obv"},
            {"type": "roc"},
            {"type": "cci"},
        ]
        cols = engine.get_feature_columns(specs)
        expected = [
            "ret_1h",
            "ret_4h",
            "rsi",
            "trend_10",
            "trend_50",
            "trend_strength",
            "vol",
            "vol_norm",
            "vol_ratio",
            "bb_pos",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "atr",
            "stoch_k",
            "stoch_d",
            "obv",
            "roc",
            "cci",
        ]
        assert cols == expected

    def test_get_feature_columns_empty(self, engine):
        cols = engine.get_feature_columns([])
        assert cols == []
