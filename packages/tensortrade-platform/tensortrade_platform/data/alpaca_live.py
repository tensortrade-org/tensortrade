"""Real-time crypto bar stream from Alpaca Markets.

Connects to the Alpaca ``CryptoDataStream`` WebSocket for live bar updates
and maintains a rolling DataFrame buffer for the agent's observation window.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

import pandas as pd

from tensortrade_platform.data.alpaca_crypto import (
    TIMEFRAME_MAP,
    AlpacaCryptoData,
    AlpacaCryptoError,
)

logger = logging.getLogger(__name__)

OnBarCallback = Callable[[pd.DataFrame], None]


class AlpacaLiveStream:
    """Stream live crypto bars from Alpaca and maintain a rolling buffer.

    On startup the stream fetches historical bars via
    :class:`~tensortrade.data.alpaca_crypto.AlpacaCryptoData` to pre-fill
    the observation window, then subscribes to the Alpaca
    ``CryptoDataStream`` WebSocket for real-time updates.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTC/USD"``.
    timeframe : str
        Bar timeframe (must be in ``TIMEFRAME_MAP``).
    api_key : str
        Alpaca API key. Falls back to ``ALPACA_API_KEY`` env var.
    secret_key : str
        Alpaca secret key. Falls back to ``ALPACA_SECRET_KEY`` env var.
    buffer_size : int
        Maximum number of bars to keep in the rolling buffer.
    on_bar : OnBarCallback | None
        Callback invoked with the full buffer DataFrame whenever a new bar
        arrives.
    """

    def __init__(
        self,
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        api_key: str = "",
        secret_key: str = "",
        buffer_size: int = 1000,
        on_bar: OnBarCallback | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.buffer_size = buffer_size
        self.on_bar = on_bar

        if timeframe not in TIMEFRAME_MAP:
            supported = ", ".join(sorted(TIMEFRAME_MAP.keys()))
            raise AlpacaCryptoError(
                f"Unsupported timeframe: {timeframe}. Supported: {supported}"
            )

        self._buffer: pd.DataFrame = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        self._stream: object | None = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_backoff_sec = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> pd.DataFrame:
        """Current bar buffer (read-only copy)."""
        return self._buffer.copy()

    @property
    def latest_price(self) -> float:
        """Most recent close price, or 0.0 if buffer is empty."""
        if self._buffer.empty:
            return 0.0
        return float(self._buffer["close"].iloc[-1])

    async def start(self) -> None:
        """Fetch historical bars, then connect to the live WebSocket."""
        self._running = True
        self._prefill_buffer()
        await self._connect()

    async def stop(self) -> None:
        """Gracefully disconnect from the stream."""
        self._running = False
        if self._stream is not None:
            try:
                await self._stream.close()
            except Exception:
                logger.debug("Error closing stream", exc_info=True)
        logger.info("AlpacaLiveStream stopped for %s", self.symbol)

    # ------------------------------------------------------------------
    # Internal: historical prefill
    # ------------------------------------------------------------------

    def _prefill_buffer(self) -> None:
        """Use the historical REST client to seed the buffer."""
        fetcher = AlpacaCryptoData()
        try:
            df = fetcher.fetch(
                symbol=self.symbol,
                timeframe=self.timeframe,
            )
            # Keep only the tail to stay within buffer_size
            if len(df) > self.buffer_size:
                df = df.tail(self.buffer_size).reset_index(drop=True)
            self._buffer = df
            logger.info(
                "Pre-filled buffer with %d historical %s bars for %s",
                len(df),
                self.timeframe,
                self.symbol,
            )
        except AlpacaCryptoError:
            logger.warning(
                "Failed to prefill buffer for %s — starting empty",
                self.symbol,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal: WebSocket connection + reconnect
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Connect to Alpaca CryptoDataStream with reconnection logic."""
        try:
            from alpaca.data.live.crypto import CryptoDataStream
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for live streaming. Install with: uv pip install alpaca-py"
            ) from exc

        while self._running:
            try:
                self._stream = CryptoDataStream(
                    self.api_key,
                    self.secret_key,
                )
                self._stream.subscribe_bars(self._handle_bar, self.symbol)
                logger.info(
                    "Connecting to Alpaca CryptoDataStream for %s (%s)",
                    self.symbol,
                    self.timeframe,
                )
                self._reconnect_attempts = 0
                await self._stream._run_forever()
            except Exception:
                if not self._running:
                    break
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    logger.error(
                        "Max reconnection attempts (%d) reached — giving up",
                        self._max_reconnect_attempts,
                    )
                    break
                backoff = self._base_backoff_sec * (2 ** (self._reconnect_attempts - 1))
                backoff = min(backoff, 60.0)
                logger.warning(
                    "Stream disconnected, reconnecting in %.1fs (attempt %d/%d)",
                    backoff,
                    self._reconnect_attempts,
                    self._max_reconnect_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)

    async def _handle_bar(self, bar: object) -> None:
        """Process an incoming bar from the WebSocket stream."""
        try:
            new_row = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(bar.timestamp).tz_localize(None)
                        if hasattr(bar, "timestamp")
                        else pd.Timestamp.utcnow(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                ]
            )
            self._buffer = pd.concat([self._buffer, new_row], ignore_index=True)
            # Trim to buffer_size
            if len(self._buffer) > self.buffer_size:
                self._buffer = self._buffer.tail(self.buffer_size).reset_index(
                    drop=True
                )

            logger.debug(
                "Bar received for %s: close=%.2f  buffer_len=%d",
                self.symbol,
                float(bar.close),
                len(self._buffer),
            )

            if self.on_bar is not None:
                self.on_bar(self._buffer.copy())
        except Exception:
            logger.exception("Error processing bar for %s", self.symbol)
