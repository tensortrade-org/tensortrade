"""Fetch OHLCV crypto data from Alpaca Markets.

Uses the alpaca-py SDK for free crypto bar data.
API keys are optional (lower rate limits without auth).
"""

import logging
import os
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
    "1w": "1Week",
}

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


class AlpacaCryptoError(Exception):
    """Raised when Alpaca crypto data fetch fails."""


class AlpacaCryptoData:
    """Fetch OHLCV crypto data from Alpaca Markets.

    Reads ALPACA_API_KEY and ALPACA_SECRET_KEY from environment.
    Falls back to no-auth client if keys are not set (lower rate limits).
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    def fetch(
        self,
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a crypto pair.

        Parameters
        ----------
        symbol : str
            Crypto pair, e.g. "BTC/USD", "ETH/USD".
        timeframe : str
            Candle timeframe: "1m", "5m", "15m", "1h", "4h", "1d", "1w".
        start_date : str
            Start date as "YYYY-MM-DD". Defaults to 2 years ago.
        end_date : str
            End date as "YYYY-MM-DD". Defaults to now.

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume.
            Sorted ascending by date, with data quality applied.
        """
        try:
            from alpaca.data.historical.crypto import (
                CryptoBarsRequest,
                CryptoHistoricalDataClient,
            )
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as e:
            raise AlpacaCryptoError(
                "alpaca-py is not installed. Install with: uv pip install -e '.[alpaca]'"
            ) from e

        # Build timeframe
        tf = self._build_timeframe(timeframe, TimeFrame, TimeFrameUnit)

        # Date range
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        else:
            start = datetime.now(UTC) - timedelta(days=730)

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        else:
            end = datetime.now(UTC)

        # Build client (with or without auth)
        if self.api_key and self.secret_key:
            client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        else:
            client = CryptoHistoricalDataClient()

        logger.info(
            "Fetching %s %s from Alpaca (%s to %s)",
            symbol,
            timeframe,
            start.date(),
            end.date(),
        )

        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )
            bars = client.get_crypto_bars(request)
        except Exception as e:
            raise AlpacaCryptoError(f"Failed to fetch {symbol} from Alpaca: {e}") from e

        df = bars.df
        if df.empty:
            raise AlpacaCryptoError(f"No data returned for {symbol} ({timeframe})")

        df = self._normalize(df, symbol)
        df = self._apply_data_quality(df, timeframe)
        return df

    @staticmethod
    def _build_timeframe(
        timeframe: str,
        tf_class: type,
        unit_class: type,
    ) -> object:
        """Convert internal timeframe string to Alpaca TimeFrame object."""
        mapping: dict[str, tuple[int, str]] = {
            "1m": (1, "Minute"),
            "5m": (5, "Minute"),
            "15m": (15, "Minute"),
            "1h": (1, "Hour"),
            "4h": (4, "Hour"),
            "1d": (1, "Day"),
            "1w": (1, "Week"),
        }
        if timeframe not in mapping:
            supported = ", ".join(sorted(mapping.keys()))
            raise AlpacaCryptoError(
                f"Unsupported timeframe: {timeframe}. Supported: {supported}"
            )
        amount, unit_name = mapping[timeframe]
        unit = getattr(unit_class, unit_name)
        return tf_class(amount, unit)

    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize Alpaca bars DataFrame to standard OHLCV format."""
        # Alpaca returns a MultiIndex (symbol, timestamp) — drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        df = df.reset_index()

        # Rename timestamp column to date
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})

        # Strip timezone
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        # Select and ensure OHLCV columns
        ohlcv_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        for col in ohlcv_map:
            if col not in df.columns:
                raise AlpacaCryptoError(
                    f"Missing column '{col}' in Alpaca response for {symbol}"
                )

        df = df[["date", "open", "high", "low", "close", "volume"]].copy()

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _apply_data_quality(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Sort, deduplicate, detect gaps, and forward-fill missing data."""
        if df.empty:
            return df

        # Sort ascending by date
        df = df.sort_values("date").reset_index(drop=True)

        # Remove duplicate timestamps
        before = len(df)
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        dupes = before - len(df)
        if dupes > 0:
            logger.warning("Removed %d duplicate timestamps", dupes)

        # Drop rows with NaN in critical columns
        critical = ["open", "high", "low", "close"]
        na_before = len(df)
        df = df.dropna(subset=critical, how="all").reset_index(drop=True)
        na_dropped = na_before - len(df)
        if na_dropped > 0:
            logger.warning("Dropped %d rows with all-NaN OHLC", na_dropped)

        # Forward-fill remaining NaN values in OHLCV
        ohlcv = ["open", "high", "low", "close", "volume"]
        df[ohlcv] = df[ohlcv].ffill()
        df[ohlcv] = df[ohlcv].bfill()

        # Detect gaps
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe)
        if tf_seconds and len(df) > 1:
            diffs = df["date"].diff().dt.total_seconds().dropna()
            expected = float(tf_seconds)
            gaps = diffs[diffs > expected * 1.5]
            if len(gaps) > 0:
                pct = len(gaps) / len(df) * 100
                logger.info(
                    "Detected %d gaps (%.1f%% of %d candles) in %s data",
                    len(gaps),
                    pct,
                    len(df),
                    timeframe,
                )

        # Drop inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df[ohlcv] = df[ohlcv].ffill().bfill()

        return df
