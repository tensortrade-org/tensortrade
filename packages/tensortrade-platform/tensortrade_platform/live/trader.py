"""Core live/paper trading engine.

Loads a trained RLlib policy, subscribes to real-time bars via
:class:`~tensortrade_platform.data.alpaca_live.AlpacaLiveStream`, computes
features on each new bar, runs inference, and executes trades through
Alpaca.  All state is persisted via :class:`LiveTradingStore`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tensortrade.oms.orders import TradeSide
from tensortrade.oms.services.execution.alpaca import AlpacaExecutionService
from tensortrade_platform.data.alpaca_live import AlpacaLiveStream
from tensortrade_platform.live.account_sync import AccountSync, PositionInfo
from tensortrade_platform.live.config import LiveTradingConfig
from tensortrade_platform.live.store import LiveTradingStore
from tensortrade_platform.training.feature_engine import FeatureEngine

if TYPE_CHECKING:
    from tensortrade_platform.api.server import ConnectionManager

logger = logging.getLogger(__name__)

# BSH action labels (matches training BSH scheme)
_ACTION_LABELS = {0: "hold", 1: "buy", 2: "sell"}


class LiveTrader:
    """Orchestrates live/paper trading for a single symbol.

    Runs as an asyncio task within the FastAPI process.  The lifecycle is:

    1. ``start()`` — load policy, warm up data, begin streaming bars.
    2. On each new bar the internal callback runs inference, executes
       trades, logs experience tuples, and broadcasts WebSocket messages.
    3. ``stop()`` — tear down the stream and mark the session as stopped.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

        # Set during start()
        self._config: LiveTradingConfig | None = None
        self._manager: ConnectionManager | None = None
        self._session_id: str = ""
        self._store: LiveTradingStore | None = None
        self._stream: AlpacaLiveStream | None = None
        self._execution: AlpacaExecutionService | None = None
        self._account_sync: AccountSync | None = None
        self._policy: object | None = None

        # Feature engine
        self._feature_engine = FeatureEngine()
        self._feature_cols: list[str] = []

        # Portfolio tracking
        self._position: int = 0  # 0=cash, 1=asset (BSH scheme)
        self._position_qty: float = 0.0
        self._entry_price: float = 0.0
        self._initial_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._pnl: float = 0.0
        self._max_drawdown_pct: float = 0.0

        # Step tracking
        self._step: int = 0
        self._total_trades: int = 0
        self._last_price: float = 0.0
        self._prev_obs: np.ndarray | None = None

        # Queue for bar events (stream callback -> async loop)
        self._bar_queue: asyncio.Queue[pd.DataFrame] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(
        self,
        config: LiveTradingConfig,
        manager: ConnectionManager,
    ) -> None:
        """Initialize services, load policy, and begin the trading loop.

        Parameters
        ----------
        config : LiveTradingConfig
            Validated session configuration.
        manager : ConnectionManager
            WebSocket connection manager for broadcasting updates.
        """
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        self._config = config
        self._manager = manager
        self._running = True

        # --- Store ---
        self._store = LiveTradingStore()

        # --- Alpaca services ---
        self._execution = AlpacaExecutionService(
            config.api_key,
            config.secret_key,
            paper=config.is_paper,
        )
        self._account_sync = AccountSync(
            config.api_key,
            config.secret_key,
            paper=config.is_paper,
        )

        # --- Load policy (blocking, run in executor) ---
        loop = asyncio.get_event_loop()
        self._policy = await loop.run_in_executor(
            None,
            self._load_policy,
            config.checkpoint_path,
        )

        # --- Feature columns ---
        if config.feature_specs:
            self._feature_cols = self._feature_engine.get_feature_columns(
                config.feature_specs,  # type: ignore[arg-type]
            )
        else:
            self._feature_cols = []

        # --- Fetch initial equity and sync existing positions ---
        snap = await loop.run_in_executor(None, self._account_sync.snapshot)
        self._current_equity = snap.equity

        # Detect existing position and capture entry price / unrealized PnL
        matched_pos: PositionInfo | None = None
        for pos in snap.positions:
            if pos.symbol.replace("/", "") == config.symbol.replace("/", ""):
                matched_pos = pos
                self._position = 1
                self._position_qty = pos.qty
                self._entry_price = pos.avg_entry_price
                self._account_sync.set_local_position(config.symbol, pos.qty)
                break

        if matched_pos is not None:
            # Subtract unrealized PnL so _pnl immediately reflects existing gains/losses
            self._initial_equity = snap.equity - matched_pos.unrealized_pl
            self._pnl = matched_pos.unrealized_pl
            logger.info(
                "Synced existing position: %s qty=%.4f entry=%.2f unrealized_pl=%.2f",
                config.symbol,
                matched_pos.qty,
                matched_pos.avg_entry_price,
                matched_pos.unrealized_pl,
            )
        else:
            self._initial_equity = snap.equity

        self._peak_equity = snap.equity

        # --- Create DB session ---
        self._session_id = self._store.create_session(
            experiment_id=config.experiment_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
            config={
                "checkpoint_path": config.checkpoint_path,
                "feature_specs": config.feature_specs,
                "window_size": config.window_size,
                "max_position_size_usd": config.max_position_size_usd,
                "max_drawdown_pct": config.max_drawdown_pct,
                "paper": config.is_paper,
            },
            initial_equity=self._initial_equity,
        )

        # --- Live stream ---
        self._stream = AlpacaLiveStream(
            symbol=config.symbol,
            timeframe=config.timeframe,
            api_key=config.api_key,
            secret_key=config.secret_key,
            buffer_size=max(config.window_size * 10, 500),
            on_bar=self._on_bar_callback,
        )

        # --- Broadcast initial status ---
        await self._broadcast_status("running")

        # --- Launch the main loop as a background task ---
        self._task = asyncio.create_task(self._run())

    async def resume(
        self,
        session_row: dict,
        config: LiveTradingConfig,
        manager: ConnectionManager,
    ) -> None:
        """Resume an existing session from DB state + Alpaca account.

        Parameters
        ----------
        session_row : dict
            Row from ``live_sessions`` (as returned by
            :meth:`LiveTradingStore.get_running_session`).
        config : LiveTradingConfig
            Reconstructed config with checkpoint path, feature specs, etc.
        manager : ConnectionManager
            WebSocket connection manager for broadcasting updates.
        """
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        self._config = config
        self._manager = manager
        self._running = True

        # --- Store ---
        self._store = LiveTradingStore()

        # --- Reuse existing session ID ---
        self._session_id = session_row["id"]

        # --- Restore counters from DB ---
        self._step = session_row.get("total_bars") or 0
        self._total_trades = session_row.get("total_trades") or 0
        self._pnl = session_row.get("pnl") or 0.0
        self._max_drawdown_pct = session_row.get("max_drawdown_pct") or 0.0
        self._initial_equity = session_row.get("initial_equity") or 0.0

        # --- Alpaca services ---
        self._execution = AlpacaExecutionService(
            config.api_key,
            config.secret_key,
            paper=config.is_paper,
        )
        self._account_sync = AccountSync(
            config.api_key,
            config.secret_key,
            paper=config.is_paper,
        )

        # --- Load policy (blocking, run in executor) ---
        loop = asyncio.get_event_loop()
        self._policy = await loop.run_in_executor(
            None,
            self._load_policy,
            config.checkpoint_path,
        )

        # --- Feature columns ---
        if config.feature_specs:
            self._feature_cols = self._feature_engine.get_feature_columns(
                config.feature_specs,  # type: ignore[arg-type]
            )
        else:
            self._feature_cols = []

        # --- Fetch current equity and sync positions from Alpaca ---
        snap = await loop.run_in_executor(None, self._account_sync.snapshot)
        self._current_equity = snap.equity

        for pos in snap.positions:
            if pos.symbol.replace("/", "") == config.symbol.replace("/", ""):
                self._position = 1
                self._position_qty = pos.qty
                self._entry_price = pos.avg_entry_price
                self._account_sync.set_local_position(config.symbol, pos.qty)
                break

        # Restore peak equity from DB (or derive from current)
        db_peak = session_row.get("peak_equity")
        if db_peak and db_peak > 0:
            self._peak_equity = max(db_peak, snap.equity)
        else:
            self._peak_equity = max(self._initial_equity, snap.equity)

        # Recalculate PnL and drawdown from live Alpaca equity
        # (the DB values may be stale if the session didn't hit a save checkpoint)
        if self._initial_equity > 0:
            self._pnl = snap.equity - self._initial_equity
            if self._peak_equity > 0:
                dd = ((self._peak_equity - snap.equity) / self._peak_equity) * 100
                self._max_drawdown_pct = max(self._max_drawdown_pct, dd)

        logger.info(
            "Resuming session %s at step=%d trades=%d position=%d "
            "equity=%.2f pnl=%.2f drawdown=%.2f%%",
            self._session_id,
            self._step,
            self._total_trades,
            self._position,
            self._current_equity,
            self._pnl,
            self._max_drawdown_pct,
        )

        # --- Live stream ---
        self._stream = AlpacaLiveStream(
            symbol=config.symbol,
            timeframe=config.timeframe,
            api_key=config.api_key,
            secret_key=config.secret_key,
            buffer_size=max(config.window_size * 10, 500),
            on_bar=self._on_bar_callback,
        )

        # --- Broadcast initial status ---
        await self._broadcast_status("running")

        # --- Launch the main loop as a background task ---
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Gracefully stop the trading session (user-initiated)."""
        self._running = False

        if self._stream is not None:
            await self._stream.stop()

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Finalize session in DB
        if self._store and self._session_id:
            self._store.update_session(
                self._session_id,
                status="stopped",
                final_equity=self._current_equity,
                total_trades=self._total_trades,
                total_bars=self._step,
                pnl=self._pnl,
                max_drawdown_pct=self._max_drawdown_pct,
                peak_equity=self._peak_equity,
            )

        # Cleanup policy / Ray
        if self._policy is not None and hasattr(self._policy, "stop"):
            try:
                self._policy.stop()
            except Exception:
                logger.debug("Failed to stop policy cleanly", exc_info=True)
        from tensortrade_platform.ray_manager import ray_manager

        ray_manager.release("live_trading")

        await self._broadcast_status("stopped")
        logger.info("LiveTrader stopped for session %s", self._session_id)

    async def pause(self) -> None:
        """Suspend trading without finalizing the DB session.

        Called during server shutdown so the session remains ``status='running'``
        in the database and can be auto-resumed on next startup.
        """
        self._running = False

        # Save latest state before shutting down
        if self._store and self._session_id:
            self._store.update_session(
                self._session_id,
                total_bars=self._step,
                total_trades=self._total_trades,
                pnl=self._pnl,
                max_drawdown_pct=self._max_drawdown_pct,
                peak_equity=self._peak_equity,
            )

        if self._stream is not None:
            await self._stream.stop()

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Cleanup policy / Ray
        if self._policy is not None and hasattr(self._policy, "stop"):
            try:
                self._policy.stop()
            except Exception:
                logger.debug("Failed to stop policy cleanly", exc_info=True)
        from tensortrade_platform.ray_manager import ray_manager

        ray_manager.release("live_trading")

        logger.info(
            "LiveTrader paused for session %s (will resume on restart)",
            self._session_id,
        )

    def get_initial_bars(self) -> list[dict[str, object]]:
        """Return buffered bars as a list of LiveBar-shaped dicts."""
        if self._stream is None:
            return []
        buf = self._stream.buffer
        if buf.empty:
            return []
        result: list[dict[str, object]] = []
        for _, row in buf.iterrows():
            ts = (
                int(pd.Timestamp(row["date"]).timestamp()) if "date" in row.index else 0
            )
            result.append(
                {
                    "timestamp": ts,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                }
            )
        return result

    def get_session_trades(self) -> list[dict[str, object]]:
        """Return persisted trades for the current session."""
        if not self._store or not self._session_id:
            return []
        trades = self._store.get_session_trades(self._session_id)
        result: list[dict[str, object]] = []
        for t in trades:
            ts = int(pd.Timestamp(t.timestamp).timestamp()) if t.timestamp else 0
            result.append(
                {
                    "step": t.step,
                    "timestamp": ts,
                    "side": t.side,
                    "symbol": t.symbol,
                    "price": t.price,
                    "size": t.size,
                    "commission": t.commission,
                    "alpaca_order_id": t.alpaca_order_id,
                }
            )
        return result

    def get_status(self) -> dict[str, object]:
        """Return a snapshot of current trading state."""
        config = self._config
        return {
            "running": self._running,
            "session_id": self._session_id,
            "experiment_id": config.experiment_id if config else "",
            "symbol": config.symbol if config else "",
            "timeframe": config.timeframe if config else "",
            "paper": config.is_paper if config else True,
            "equity": self._current_equity,
            "initial_equity": self._initial_equity,
            "pnl": self._pnl,
            "pnl_pct": (self._pnl / self._initial_equity * 100)
            if self._initial_equity > 0
            else 0.0,
            "position": self._position,
            "position_qty": self._position_qty,
            "entry_price": self._entry_price,
            "total_trades": self._total_trades,
            "total_bars": self._step,
            "max_drawdown_pct": self._max_drawdown_pct,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _broadcast_bars_history(self) -> None:
        """Send all buffered bars as a single history message."""
        if self._manager is None or self._stream is None:
            return
        bars = self.get_initial_bars()
        if bars:
            await self._manager.broadcast_to_dashboards(
                {"type": "live_bars_history", "bars": bars}
            )
            logger.info("Broadcast %d initial bars to dashboard", len(bars))

    async def _run(self) -> None:
        """Main coroutine: start stream, then process bars from queue."""
        try:
            assert self._stream is not None
            # Start stream in background (pre-fills buffer, then connects WS)
            stream_task = asyncio.create_task(self._stream.start())

            # Wait for the prefill to complete (it's synchronous inside start())
            # then broadcast the historical bars so the chart isn't blank.
            for _ in range(60):
                await asyncio.sleep(0.5)
                if self._stream.latest_price > 0:
                    break
            await self._broadcast_bars_history()

            # Process bars from the queue
            while self._running:
                try:
                    buffer = await asyncio.wait_for(
                        self._bar_queue.get(),
                        timeout=5.0,
                    )
                except TimeoutError:
                    # No bar received — check if still running
                    continue

                await self._process_bar(buffer)

            # When stopping, also cancel the stream task
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

        except asyncio.CancelledError:
            logger.info("LiveTrader task cancelled")
        except Exception:
            logger.exception("LiveTrader loop error")
            if self._store and self._session_id:
                self._store.update_session(self._session_id, status="error")
            await self._broadcast_status("error")

    def _on_bar_callback(self, buffer: pd.DataFrame) -> None:
        """Called by AlpacaLiveStream from the WS handler (sync)."""
        try:
            self._bar_queue.put_nowait(buffer)
        except asyncio.QueueFull:
            logger.warning("Bar queue full, dropping bar")

    # ------------------------------------------------------------------
    # Per-bar processing
    # ------------------------------------------------------------------

    async def _process_bar(self, buffer: pd.DataFrame) -> None:
        """Process a single new bar: features -> inference -> execute."""
        assert self._config is not None
        assert self._store is not None
        config = self._config

        self._step += 1

        # --- Extract latest bar for broadcast ---
        if buffer.empty:
            return
        latest = buffer.iloc[-1]
        price = float(latest["close"])
        self._last_price = price

        await self._broadcast_bar(latest)

        # --- Compute features on rolling buffer ---
        if config.feature_specs:
            featured = self._feature_engine.compute(
                buffer.copy(),
                config.feature_specs,  # type: ignore[arg-type]
            )
        else:
            featured = buffer.copy()

        # --- Build observation vector ---
        obs = self._build_observation(featured, config.window_size)
        if obs is None:
            logger.debug("Not enough data for observation (step %d)", self._step)
            return

        # --- Run inference ---
        action = await asyncio.get_event_loop().run_in_executor(
            None,
            self._compute_action,
            obs,
        )
        action_label = _ACTION_LABELS.get(action, "hold")

        await self._broadcast_action(action, action_label, price)

        # --- Execute trade if needed ---
        await self._maybe_execute(action, price, config)

        # --- Compute reward (simple PnL since last bar) ---
        reward = self._compute_reward(price)

        # --- Store experience ---
        if self._prev_obs is not None:
            self._store.log_experience(
                session_id=self._session_id,
                step=self._step,
                observation=self._prev_obs.tobytes(),
                action=action,
                reward=reward,
                next_observation=obs.tobytes(),
                done=False,
                symbol=config.symbol,
                price=price,
            )
        self._prev_obs = obs

        # --- Sync portfolio periodically (every 10 bars) ---
        if self._step % 10 == 0:
            await self._sync_portfolio()

        # --- Check drawdown safety ---
        if self._max_drawdown_pct >= config.max_drawdown_pct:
            logger.warning(
                "Max drawdown (%.1f%%) exceeded limit (%.1f%%) — stopping",
                self._max_drawdown_pct,
                config.max_drawdown_pct,
            )
            self._running = False

        # Update session in DB periodically
        if self._step % 5 == 0 and self._store:
            self._store.update_session(
                self._session_id,
                total_bars=self._step,
                total_trades=self._total_trades,
                pnl=self._pnl,
                max_drawdown_pct=self._max_drawdown_pct,
                peak_equity=self._peak_equity,
            )

    def _build_observation(
        self,
        featured: pd.DataFrame,
        window_size: int,
    ) -> np.ndarray | None:
        """Build a flat observation vector from the last window_size rows."""
        if len(featured) < window_size:
            return None

        # Select feature columns that exist in the dataframe
        cols = [c for c in self._feature_cols if c in featured.columns]
        if not cols:
            # Fallback: use all non-OHLCV columns
            ohlcv = {"date", "open", "high", "low", "close", "volume"}
            cols = [c for c in featured.columns if c not in ohlcv]

        if not cols:
            return None

        window = featured[cols].iloc[-window_size:]
        obs = window.to_numpy(dtype=np.float32).flatten()
        # Replace NaN with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _compute_action(self, obs: np.ndarray) -> int:
        """Run the policy on a single observation."""
        raw = self._policy.compute_single_action(obs, explore=False)  # type: ignore[union-attr]
        return _coerce_action(raw)

    async def _maybe_execute(
        self,
        action: int,
        price: float,
        config: LiveTradingConfig,
    ) -> dict[str, object] | None:
        """Execute a trade if the action requires one.

        BSH scheme: 0=hold, 1=buy (if in cash), 2=sell (if in asset).
        """
        assert self._execution is not None
        assert self._store is not None

        if action == 1 and self._position == 0:
            # Buy: compute qty from max position size and current price
            notional = config.max_position_size_usd
            if notional <= 0 or price <= 0:
                return None

            qty = notional / price
            fill = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._execution.execute(
                    symbol=config.symbol,
                    side=TradeSide.BUY,
                    qty=qty,
                ),
            )
            if fill is not None:
                self._position = 1
                self._position_qty = fill.filled_qty
                self._entry_price = fill.filled_avg_price
                self._total_trades += 1
                if self._account_sync:
                    self._account_sync.set_local_position(
                        config.symbol,
                        fill.filled_qty,
                    )
                trade_id = self._store.log_live_trade(
                    session_id=self._session_id,
                    step=self._step,
                    side="buy",
                    symbol=config.symbol,
                    price=fill.filled_avg_price,
                    size=fill.filled_qty,
                    commission=fill.commission,
                    alpaca_order_id=fill.order_id,
                )
                await self._broadcast_trade(
                    "buy",
                    fill.filled_avg_price,
                    fill.filled_qty,
                    fill.order_id,
                    fill.commission,
                )
                return {"trade_id": trade_id, "fill": fill}

        elif action == 2 and self._position == 1:
            # Sell: liquidate entire position
            qty = self._position_qty
            if qty <= 0:
                return None

            fill = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._execution.execute(
                    symbol=config.symbol,
                    side=TradeSide.SELL,
                    qty=qty,
                ),
            )
            if fill is not None:
                self._position = 0
                self._position_qty = 0.0
                self._entry_price = 0.0
                self._total_trades += 1
                if self._account_sync:
                    self._account_sync.set_local_position(config.symbol, 0.0)
                trade_id = self._store.log_live_trade(
                    session_id=self._session_id,
                    step=self._step,
                    side="sell",
                    symbol=config.symbol,
                    price=fill.filled_avg_price,
                    size=fill.filled_qty,
                    commission=fill.commission,
                    alpaca_order_id=fill.order_id,
                )
                await self._broadcast_trade(
                    "sell",
                    fill.filled_avg_price,
                    fill.filled_qty,
                    fill.order_id,
                    fill.commission,
                )
                return {"trade_id": trade_id, "fill": fill}

        return None

    def _compute_reward(self, current_price: float) -> float:
        """Simple PnL reward: price change since last bar."""
        if self._position == 0 or self._last_price == 0:
            return 0.0
        return (current_price - self._last_price) / self._last_price

    async def _sync_portfolio(self) -> None:
        """Sync equity/drawdown from Alpaca."""
        if self._account_sync is None:
            return

        loop = asyncio.get_event_loop()
        snap = await loop.run_in_executor(None, self._account_sync.snapshot)

        self._current_equity = snap.equity
        if snap.equity > self._peak_equity:
            self._peak_equity = snap.equity
        if self._peak_equity > 0:
            dd = ((self._peak_equity - snap.equity) / self._peak_equity) * 100
            self._max_drawdown_pct = max(self._max_drawdown_pct, dd)
        self._pnl = snap.equity - self._initial_equity

        await self._broadcast_portfolio()

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_policy(checkpoint_path: str) -> object:
        """Load an RLlib policy from a checkpoint directory.

        Ray lifecycle is managed by :mod:`tensortrade_platform.ray_manager`.
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        from ray.rllib.policy.policy import Policy

        from tensortrade_platform.ray_manager import ray_manager

        ray_manager.acquire("live_trading")

        policies = Policy.from_checkpoint(checkpoint_path)
        policy = policies.get("default_policy")
        if policy is None:
            raise ValueError(
                f"No default_policy found in checkpoint: {checkpoint_path}",
            )
        return policy

    # ------------------------------------------------------------------
    # WebSocket broadcasts
    # ------------------------------------------------------------------

    async def _broadcast_status(self, state: str) -> None:
        if self._manager is None:
            return
        pnl_pct = (
            (self._pnl / self._initial_equity * 100)
            if self._initial_equity > 0
            else 0.0
        )
        await self._manager.broadcast_to_dashboards(
            {
                "type": "live_status",
                "state": state,
                "session_id": self._session_id,
                "symbol": self._config.symbol if self._config else "",
                "equity": self._current_equity,
                "pnl": self._pnl,
                "pnl_pct": pnl_pct,
                "position": "asset" if self._position == 1 else "cash",
                "entry_price": self._entry_price,
                "total_bars": self._step,
                "total_trades": self._total_trades,
                "drawdown_pct": self._max_drawdown_pct,
                "source": "live",
            }
        )

    async def _broadcast_bar(self, row: pd.Series) -> None:
        if self._manager is None:
            return
        ts = (
            int(pd.Timestamp(row["date"]).timestamp())
            if "date" in row.index
            else self._step
        )
        await self._manager.broadcast_to_dashboards(
            {
                "type": "live_bar",
                "step": self._step,
                "timestamp": ts,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
                "source": "live",
            }
        )

    async def _broadcast_action(
        self,
        action: int,
        label: str,
        price: float,
    ) -> None:
        if self._manager is None:
            return
        await self._manager.broadcast_to_dashboards(
            {
                "type": "live_action",
                "step": self._step,
                "action": action,
                "action_label": label,
                "price": price,
                "position": self._position,  # 0=cash, 1=asset
                "timestamp": int(time.time()),
            }
        )

    async def _broadcast_trade(
        self,
        side: str,
        price: float,
        size: float,
        alpaca_order_id: str,
        commission: float,
    ) -> None:
        if self._manager is None:
            return
        await self._manager.broadcast_to_dashboards(
            {
                "type": "live_trade",
                "step": self._step,
                "timestamp": int(time.time()),
                "side": side,
                "price": price,
                "size": size,
                "alpaca_order_id": alpaca_order_id,
                "commission": commission,
                "entry_price": self._entry_price,
                "source": "live",
            }
        )

    async def _broadcast_portfolio(self) -> None:
        if self._manager is None:
            return
        pnl_pct = (
            (self._pnl / self._initial_equity * 100)
            if self._initial_equity > 0
            else 0.0
        )
        await self._manager.broadcast_to_dashboards(
            {
                "type": "live_portfolio",
                "equity": self._current_equity,
                "cash": self._current_equity - (self._position_qty * self._last_price),
                "position_value": self._position_qty * self._last_price,
                "pnl": self._pnl,
                "pnl_pct": pnl_pct,
                "drawdown_pct": self._max_drawdown_pct,
                "entry_price": self._entry_price,
                "source": "live",
            }
        )


def _coerce_action(action_value: object) -> int:
    """Normalize RLlib action outputs to a plain integer.

    Matches ``InferenceRunner._coerce_action`` logic.
    """
    value = action_value
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        value = value[0] if value else 0
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return int(value)  # type: ignore[arg-type]
