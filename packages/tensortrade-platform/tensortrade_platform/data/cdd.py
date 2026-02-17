"""Fetch OHLCV crypto data from https://www.cryptodatadownload.com.

Uses standard crypto convention: base=BTC, quote=USD means BTC/USD pair.
Supported exchanges: Bitfinex, Bitstamp, Binance (USDT), Gemini.
"""

import logging
import ssl
import urllib.error
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Exchanges known to have data on CDD and their quirks
SUPPORTED_EXCHANGES: dict[str, dict[str, str | bool]] = {
    "Bitfinex": {"quote_override": "", "case": "lower"},
    "Bitstamp": {"quote_override": "", "case": "lower"},
    "Binance": {"quote_override": "USDT", "case": "upper"},
    "Gemini": {"quote_override": "", "case": "upper"},
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


class CryptoDataDownloadError(Exception):
    """Raised when crypto data download fails."""


class CryptoDataDownload:
    """Fetch OHLCV data from CryptoDataDownload.

    Parameters use standard crypto convention:
        base_symbol = BTC (the asset)
        quote_symbol = USD (the pricing currency)
        pair in filename = BTCUSD

    Attributes
    ----------
    url : str
        Base URL for CryptoDataDownload CSV files.
    """

    def __init__(self) -> None:
        self.url = "https://www.cryptodatadownload.com/cdd/"

    def fetch(
        self,
        exchange_name: str,
        base_symbol: str,
        quote_symbol: str,
        timeframe: str,
        include_all_volumes: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for an exchange and pair.

        Parameters
        ----------
        exchange_name : str
            Exchange name (Bitfinex, Bitstamp, Binance, Gemini).
        base_symbol : str
            Base asset, e.g. "BTC".
        quote_symbol : str
            Quote currency, e.g. "USD".
        timeframe : str
            Candle timeframe: "1m", "5m", "15m", "1h", "4h", "1d", "1w".
        include_all_volumes : bool
            If True, include both base and quote volume columns.

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume
            (plus volume_base, volume_quote if include_all_volumes).
            Sorted ascending by date, with data quality applied.
        """
        exchange = self._normalize_exchange(exchange_name)
        base = base_symbol.upper().strip()
        quote = quote_symbol.upper().strip()

        info = SUPPORTED_EXCHANGES.get(exchange)
        if info is None:
            supported = ", ".join(sorted(SUPPORTED_EXCHANGES.keys()))
            raise CryptoDataDownloadError(
                f"Exchange '{exchange_name}' is not supported. Supported: {supported}"
            )

        # Binance uses USDT, auto-convert USD -> USDT
        actual_quote = quote
        if info["quote_override"] and quote == "USD":
            actual_quote = str(info["quote_override"])

        if exchange == "Gemini":
            df = self._fetch_gemini(base, actual_quote, timeframe)
        else:
            df = self._fetch_standard(exchange, base, actual_quote, timeframe)

        df = self._normalize_columns(df, base, actual_quote, include_all_volumes)
        df = self._apply_data_quality(df, timeframe)
        return df

    def _normalize_exchange(self, name: str) -> str:
        """Match exchange name case-insensitively to known exchanges."""
        lower = name.lower().strip()
        for known in SUPPORTED_EXCHANGES:
            if known.lower() == lower:
                return known
        return name.strip()

    def _build_url(self, exchange: str, pair: str, timeframe: str) -> str:
        """Build the CDD CSV URL."""
        return f"{self.url}{exchange}_{pair}_{timeframe}.csv"

    def _download_csv(self, url: str) -> pd.DataFrame:
        """Download and parse a CSV, handling SSL and errors."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Extract filename from URL for clearer message
                filename = url.rsplit("/", 1)[-1]
                raise CryptoDataDownloadError(
                    f"Data not found: {filename}. "
                    f"Check that this exchange/pair/timeframe combination "
                    f"exists on cryptodatadownload.com"
                ) from e
            raise CryptoDataDownloadError(f"HTTP error {e.code} fetching {url}") from e
        except urllib.error.URLError as e:
            raise CryptoDataDownloadError(
                f"Network error fetching data: {e.reason}"
            ) from e

        # Skip first line (site header)
        lines = raw.split("\n", 1)
        csv_data = lines[1] if len(lines) > 1 else raw

        df = pd.read_csv(StringIO(csv_data))
        if df.empty:
            raise CryptoDataDownloadError(f"Empty dataset returned from {url}")
        return df

    def _fetch_standard(
        self, exchange: str, base: str, quote: str, timeframe: str
    ) -> pd.DataFrame:
        """Fetch from exchanges with standard CSV format (Bitfinex, Bitstamp, Binance)."""
        pair = f"{base}{quote}"
        url = self._build_url(exchange, pair, timeframe)
        logger.info("Fetching %s %s %s from CDD", exchange, pair, timeframe)
        return self._download_csv(url)

    def _fetch_gemini(self, base: str, quote: str, timeframe: str) -> pd.DataFrame:
        """Fetch from Gemini (uses lowercase prefix, different timeframe naming)."""
        tf = timeframe
        if tf.endswith("h"):
            tf = tf[:-1] + "hr"
        pair = f"{base}{quote}"
        url = f"{self.url}gemini_{pair}_{tf}.csv"
        logger.info("Fetching Gemini %s %s from CDD", pair, tf)
        return self._download_csv(url)

    def _normalize_columns(
        self,
        df: pd.DataFrame,
        base: str,
        quote: str,
        include_all_volumes: bool,
    ) -> pd.DataFrame:
        """Normalize all exchange CSV formats to consistent lowercase OHLCV columns."""
        # Lowercase all column names first
        df.columns = [c.lower().strip() for c in df.columns]

        # Drop symbol column if present
        for col in ["symbol"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Drop tradecount if present (Binance)
        if "tradecount" in df.columns:
            df = df.drop(columns=["tradecount"])

        # Handle unix timestamp → date conversion
        unix_col = None
        if "unix" in df.columns:
            unix_col = "unix"
        elif "unix timestamp" in df.columns:
            unix_col = "unix timestamp"

        if unix_col is not None:
            df[unix_col] = pd.to_numeric(df[unix_col], errors="coerce")
            df = df.dropna(subset=[unix_col])
            df[unix_col] = df[unix_col].astype(np.int64)

            # Normalize to seconds: handle microseconds (>1e15), milliseconds (>1e12)
            def _to_seconds(x: int) -> int:
                if x > 1e15:
                    return int(x // 1_000_000)
                if x > 1e12:
                    return int(x // 1_000)
                return int(x)

            df[unix_col] = df[unix_col].apply(_to_seconds)
            df["date"] = pd.to_datetime(df[unix_col], unit="s", utc=True)
            df["date"] = df["date"].dt.tz_localize(None)
            if unix_col != "date":
                df = df.drop(columns=[unix_col])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date"] = df["date"].dt.tz_localize(None)

        # Handle volume columns — find the base volume (crypto amount)
        base_lower = base.lower()
        quote_lower = quote.lower()
        volume_base_col = f"volume {base_lower}"
        volume_quote_col = f"volume {quote_lower}"

        if volume_base_col in df.columns:
            df = df.rename(columns={volume_base_col: "volume_base"})
        if volume_quote_col in df.columns:
            df = df.rename(columns={volume_quote_col: "volume_quote"})

        # If there's just a "volume" column (Gemini), use it as-is
        # Otherwise use volume_quote (USD value) as the default volume
        if "volume" not in df.columns:
            if "volume_quote" in df.columns:
                df["volume"] = df["volume_quote"]
            elif "volume_base" in df.columns:
                df["volume"] = df["volume_base"]

        # Ensure OHLCV columns exist
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise CryptoDataDownloadError(
                f"Missing required columns after normalization: {missing}. Got columns: {list(df.columns)}"
            )

        # Select output columns
        if include_all_volumes:
            extras = [c for c in ["volume_base", "volume_quote"] if c in df.columns]
            out_cols = required + [c for c in extras if c not in required]
        else:
            out_cols = required

        df = df[out_cols].copy()

        # Ensure numeric types for OHLCV
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _apply_data_quality(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
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
        # Back-fill any leading NaNs
        df[ohlcv] = df[ohlcv].bfill()

        # Detect gaps in the time series
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

        # Drop any remaining inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df[ohlcv] = df[ohlcv].ffill().bfill()

        return df
