"""
Configurable feature engineering engine.
Extracts feature computation from training scripts into a reusable, configurable engine.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# Registry of available feature types with their parameter schemas
FEATURE_CATALOG: list[dict[str, object]] = [
    {
        "type": "returns",
        "name": "Price Returns",
        "description": "Tanh-normalized percentage returns over multiple periods.",
        "params": [
            {
                "name": "periods",
                "type": "list[int]",
                "default": [1, 4, 12, 24, 48],
                "description": "Look-back periods in bars for return calculation.",
            },
            {
                "name": "normalize",
                "type": "str",
                "default": "tanh",
                "description": "Normalization method: 'tanh' or 'none'.",
            },
        ],
    },
    {
        "type": "rsi",
        "name": "RSI",
        "description": "Relative Strength Index normalized to [-1, 1].",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 100,
                "description": "RSI calculation period.",
            },
            {
                "name": "normalize",
                "type": "bool",
                "default": True,
                "description": "If True, scale RSI from [0,100] to [-1,1].",
            },
        ],
    },
    {
        "type": "sma_trend",
        "name": "SMA Trend",
        "description": "Price position relative to SMA, measuring trend direction.",
        "params": [
            {
                "name": "fast",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 200,
                "description": "Fast SMA period.",
            },
            {
                "name": "slow",
                "type": "int",
                "default": 50,
                "min": 5,
                "max": 500,
                "description": "Slow SMA period.",
            },
            {
                "name": "normalize",
                "type": "str",
                "default": "tanh",
                "description": "Normalization: 'tanh' or 'none'.",
            },
        ],
    },
    {
        "type": "trend_strength",
        "name": "Trend Strength",
        "description": "Relative difference between fast and slow SMAs.",
        "params": [
            {
                "name": "fast",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 200,
                "description": "Fast SMA period.",
            },
            {
                "name": "slow",
                "type": "int",
                "default": 50,
                "min": 5,
                "max": 500,
                "description": "Slow SMA period.",
            },
        ],
    },
    {
        "type": "volatility",
        "name": "Volatility",
        "description": "Rolling standard deviation of close price, optionally z-score normalized.",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 24,
                "min": 2,
                "max": 200,
                "description": "Rolling window for volatility.",
            },
            {
                "name": "rolling_norm_period",
                "type": "int",
                "default": 72,
                "min": 0,
                "max": 500,
                "description": "Period for rolling z-score normalization (0 = no normalization).",
            },
        ],
    },
    {
        "type": "volume_ratio",
        "name": "Volume Ratio",
        "description": "Log ratio of current volume to rolling average volume.",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 200,
                "description": "Rolling average period for volume.",
            },
        ],
    },
    {
        "type": "bollinger_position",
        "name": "Bollinger Position",
        "description": "Price position within Bollinger Bands, scaled [0, 1].",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 200,
                "description": "Bollinger Band SMA period.",
            },
            {
                "name": "std_dev",
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 4.0,
                "description": "Standard deviation multiplier for bands.",
            },
        ],
    },
    {
        "type": "macd",
        "name": "MACD",
        "description": "Moving Average Convergence Divergence with signal line and histogram, tanh-scaled.",
        "params": [
            {
                "name": "fast",
                "type": "int",
                "default": 12,
                "min": 2,
                "max": 100,
                "description": "Fast EMA period.",
            },
            {
                "name": "slow",
                "type": "int",
                "default": 26,
                "min": 5,
                "max": 200,
                "description": "Slow EMA period.",
            },
            {
                "name": "signal",
                "type": "int",
                "default": 9,
                "min": 2,
                "max": 50,
                "description": "Signal line EMA period.",
            },
        ],
    },
    {
        "type": "atr",
        "name": "ATR",
        "description": "Average True Range measuring volatility from high/low/close, rolling z-score + tanh normalized.",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 100,
                "description": "ATR calculation period.",
            },
            {
                "name": "rolling_norm_period",
                "type": "int",
                "default": 72,
                "min": 0,
                "max": 500,
                "description": "Period for rolling z-score normalization (0 = no normalization).",
            },
        ],
    },
    {
        "type": "stochastic",
        "name": "Stochastic Oscillator",
        "description": "Stochastic %K and %D using high/low/close, scaled to [-1, 1].",
        "params": [
            {
                "name": "k_period",
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 100,
                "description": "Look-back period for %K.",
            },
            {
                "name": "d_period",
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 50,
                "description": "Smoothing period for %D.",
            },
        ],
    },
    {
        "type": "obv",
        "name": "On-Balance Volume",
        "description": "Cumulative volume weighted by price direction, rolling z-score + tanh normalized.",
        "params": [
            {
                "name": "rolling_norm_period",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 500,
                "description": "Period for rolling z-score normalization.",
            },
        ],
    },
    {
        "type": "roc",
        "name": "Rate of Change",
        "description": "Percentage rate of change of close price, tanh-normalized.",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 200,
                "description": "Look-back period for rate of change.",
            },
            {
                "name": "normalize",
                "type": "str",
                "default": "tanh",
                "description": "Normalization method: 'tanh' or 'none'.",
            },
        ],
    },
    {
        "type": "cci",
        "name": "CCI",
        "description": "Commodity Channel Index using high/low/close typical price, tanh-scaled.",
        "params": [
            {
                "name": "period",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 200,
                "description": "CCI calculation period.",
            },
        ],
    },
]


FeatureComputeFn = Callable[[pd.DataFrame, dict[str, object]], pd.DataFrame]


class FeatureEngine:
    """Configurable feature computation engine.

    Computes technical indicators from OHLCV DataFrames based on
    declarative feature specifications.
    """

    AVAILABLE_FEATURES: dict[str, dict[str, object]] = {
        entry["type"]: entry
        for entry in FEATURE_CATALOG  # type: ignore[misc]
    }

    def __init__(self) -> None:
        self._compute_fns: dict[str, FeatureComputeFn] = {
            "returns": self._compute_returns,
            "rsi": self._compute_rsi,
            "sma_trend": self._compute_sma_trend,
            "trend_strength": self._compute_trend_strength,
            "volatility": self._compute_volatility,
            "volume_ratio": self._compute_volume_ratio,
            "bollinger_position": self._compute_bollinger_position,
            "macd": self._compute_macd,
            "atr": self._compute_atr,
            "stochastic": self._compute_stochastic,
            "obv": self._compute_obv,
            "roc": self._compute_roc,
            "cci": self._compute_cci,
        }

    def list_available(self) -> list[dict[str, object]]:
        """Return the feature catalog with parameter schemas."""
        return FEATURE_CATALOG

    def compute(
        self, df: pd.DataFrame, feature_specs: list[dict[str, object]]
    ) -> pd.DataFrame:
        """Compute features on a DataFrame based on specs.

        Args:
            df: DataFrame with at least 'close' column. May also have
                'open', 'high', 'low', 'volume'.
            feature_specs: List of feature spec dicts, each with a 'type'
                key and optional params.

        Returns:
            DataFrame with original columns plus computed features.
        """
        result = df.copy()
        for spec in feature_specs:
            feat_type = spec.get("type", "")
            compute_fn = self._compute_fns.get(str(feat_type))
            if compute_fn is None:
                continue
            result = compute_fn(result, spec)

        return result.bfill().ffill()

    def preview(
        self,
        df: pd.DataFrame,
        feature_specs: list[dict[str, object]],
        sample_rows: int = 100,
    ) -> dict[str, object]:
        """Compute features and return preview with stats.

        Returns dict with 'sample' (first N rows), 'stats' (per-feature
        statistics), 'feature_columns' (list of new column names), and
        'rows' (total row count).
        """
        original_cols = set(df.columns)
        result = self.compute(df, feature_specs)
        new_cols = [c for c in result.columns if c not in original_cols]

        stats: dict[str, dict[str, float]] = {}
        for col in new_cols:
            series = result[col].dropna()
            if len(series) > 0:
                stats[str(col)] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        sample = result.head(sample_rows)

        return {
            "rows": len(result),
            "feature_columns": new_cols,
            "stats": stats,
            "sample": sample.to_dict(orient="records"),
        }

    def get_feature_columns(self, feature_specs: list[dict[str, object]]) -> list[str]:
        """Return the column names that would be generated by the given specs.

        Useful for determining which columns to feed into the environment
        without actually computing features.
        """
        cols: list[str] = []
        for spec in feature_specs:
            feat_type = spec.get("type", "")
            if feat_type == "returns":
                periods = spec.get("periods", [1, 4, 12, 24, 48])
                if isinstance(periods, list):
                    for p in periods:
                        cols.append(f"ret_{p}h")
            elif feat_type == "rsi":
                cols.append("rsi")
            elif feat_type == "sma_trend":
                fast = spec.get("fast", 20)
                slow = spec.get("slow", 50)
                cols.append(f"trend_{fast}")
                cols.append(f"trend_{slow}")
            elif feat_type == "trend_strength":
                cols.append("trend_strength")
            elif feat_type == "volatility":
                cols.append("vol")
                rolling_norm = spec.get("rolling_norm_period", 72)
                if rolling_norm and isinstance(rolling_norm, int) and rolling_norm > 0:
                    cols.append("vol_norm")
            elif feat_type == "volume_ratio":
                cols.append("vol_ratio")
            elif feat_type == "bollinger_position":
                cols.append("bb_pos")
            elif feat_type == "macd":
                cols.extend(["macd_line", "macd_signal", "macd_hist"])
            elif feat_type == "atr":
                cols.append("atr")
            elif feat_type == "stochastic":
                cols.extend(["stoch_k", "stoch_d"])
            elif feat_type == "obv":
                cols.append("obv")
            elif feat_type == "roc":
                cols.append("roc")
            elif feat_type == "cci":
                cols.append("cci")
        return cols

    # --- Feature computation methods ---

    @staticmethod
    def _compute_returns(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        periods = spec.get("periods", [1, 4, 12, 24, 48])
        normalize = spec.get("normalize", "tanh")
        if isinstance(periods, list):
            for p in periods:
                col = f"ret_{p}h"
                pct = df["close"].pct_change(int(p))
                if normalize == "tanh":
                    df[col] = np.tanh(pct * 10)
                else:
                    df[col] = pct
        return df

    @staticmethod
    def _compute_rsi(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        period = int(spec.get("period", 14))  # type: ignore[arg-type]
        normalize = spec.get("normalize", True)

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        if normalize:
            df["rsi"] = (rsi - 50) / 50
        else:
            df["rsi"] = rsi
        return df

    @staticmethod
    def _compute_sma_trend(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        fast = int(spec.get("fast", 20))  # type: ignore[arg-type]
        slow = int(spec.get("slow", 50))  # type: ignore[arg-type]
        normalize = spec.get("normalize", "tanh")

        sma_fast = df["close"].rolling(fast).mean()
        sma_slow = df["close"].rolling(slow).mean()

        if normalize == "tanh":
            df[f"trend_{fast}"] = np.tanh((df["close"] - sma_fast) / sma_fast * 10)
            df[f"trend_{slow}"] = np.tanh((df["close"] - sma_slow) / sma_slow * 10)
        else:
            df[f"trend_{fast}"] = (df["close"] - sma_fast) / sma_fast
            df[f"trend_{slow}"] = (df["close"] - sma_slow) / sma_slow
        return df

    @staticmethod
    def _compute_trend_strength(
        df: pd.DataFrame, spec: dict[str, object]
    ) -> pd.DataFrame:
        fast = int(spec.get("fast", 20))  # type: ignore[arg-type]
        slow = int(spec.get("slow", 50))  # type: ignore[arg-type]

        sma_fast = df["close"].rolling(fast).mean()
        sma_slow = df["close"].rolling(slow).mean()
        df["trend_strength"] = np.tanh((sma_fast - sma_slow) / sma_slow * 20)
        return df

    @staticmethod
    def _compute_volatility(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        period = int(spec.get("period", 24))  # type: ignore[arg-type]
        rolling_norm_period = int(spec.get("rolling_norm_period", 72))  # type: ignore[arg-type]

        df["vol"] = df["close"].rolling(period).std() / df["close"]

        if rolling_norm_period > 0:
            vol_mean = df["vol"].rolling(rolling_norm_period).mean()
            vol_std = df["vol"].rolling(rolling_norm_period).std()
            df["vol_norm"] = np.tanh((df["vol"] - vol_mean) / vol_std)
        return df

    @staticmethod
    def _compute_volume_ratio(
        df: pd.DataFrame, spec: dict[str, object]
    ) -> pd.DataFrame:
        period = int(spec.get("period", 20))  # type: ignore[arg-type]

        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(period).mean()
            df["vol_ratio"] = np.log1p(df["volume"] / vol_ma)
        else:
            df["vol_ratio"] = 0.0
        return df

    @staticmethod
    def _compute_bollinger_position(
        df: pd.DataFrame, spec: dict[str, object]
    ) -> pd.DataFrame:
        period = int(spec.get("period", 20))  # type: ignore[arg-type]
        std_dev = float(spec.get("std_dev", 2.0))  # type: ignore[arg-type]

        bb_mid = df["close"].rolling(period).mean()
        bb_std = df["close"].rolling(period).std()
        lower = bb_mid - std_dev * bb_std
        band_width = 2 * std_dev * bb_std
        df["bb_pos"] = ((df["close"] - lower) / band_width).clip(0, 1)
        return df

    @staticmethod
    def _compute_macd(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        fast = int(spec.get("fast", 12))  # type: ignore[arg-type]
        slow = int(spec.get("slow", 26))  # type: ignore[arg-type]
        signal = int(spec.get("signal", 9))  # type: ignore[arg-type]

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # Normalize relative to price for scale-invariance, then tanh
        price = df["close"].replace(0, 1e-10)
        df["macd_line"] = np.tanh(macd_line / price * 100)
        df["macd_signal"] = np.tanh(macd_signal / price * 100)
        df["macd_hist"] = np.tanh(macd_hist / price * 100)
        return df

    @staticmethod
    def _compute_atr(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        period = int(spec.get("period", 14))  # type: ignore[arg-type]
        rolling_norm_period = int(spec.get("rolling_norm_period", 72))  # type: ignore[arg-type]

        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()

        if rolling_norm_period > 0:
            atr_mean = atr.rolling(rolling_norm_period).mean()
            atr_std = atr.rolling(rolling_norm_period).std()
            df["atr"] = np.tanh((atr - atr_mean) / (atr_std + 1e-10))
        else:
            df["atr"] = atr / close
        return df

    @staticmethod
    def _compute_stochastic(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        k_period = int(spec.get("k_period", 14))  # type: ignore[arg-type]
        d_period = int(spec.get("d_period", 3))  # type: ignore[arg-type]

        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        close = df["close"]

        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        denom = highest_high - lowest_low
        # %K in [0, 100], then scale to [-1, 1]
        stoch_k = ((close - lowest_low) / (denom + 1e-10)) * 100
        stoch_d = stoch_k.rolling(d_period).mean()

        df["stoch_k"] = (stoch_k - 50) / 50
        df["stoch_d"] = (stoch_d - 50) / 50
        return df

    @staticmethod
    def _compute_obv(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        rolling_norm_period = int(spec.get("rolling_norm_period", 20))  # type: ignore[arg-type]

        if "volume" in df.columns:
            direction = np.sign(df["close"].diff())
            obv = (direction * df["volume"]).cumsum()

            obv_mean = obv.rolling(rolling_norm_period).mean()
            obv_std = obv.rolling(rolling_norm_period).std()
            df["obv"] = np.tanh((obv - obv_mean) / (obv_std + 1e-10))
        else:
            df["obv"] = 0.0
        return df

    @staticmethod
    def _compute_roc(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        period = int(spec.get("period", 12))  # type: ignore[arg-type]
        normalize = spec.get("normalize", "tanh")

        roc = df["close"].pct_change(period)
        if normalize == "tanh":
            df["roc"] = np.tanh(roc * 10)
        else:
            df["roc"] = roc
        return df

    @staticmethod
    def _compute_cci(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        period = int(spec.get("period", 20))  # type: ignore[arg-type]

        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        typical_price = (high + low + df["close"]) / 3

        tp_sma = typical_price.rolling(period).mean()
        tp_mad = typical_price.rolling(period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)
        # CCI typically ranges -200 to +200; tanh(cci/200) maps to ~[-1,1]
        df["cci"] = np.tanh(cci / 200)
        return df
