"""
Persistent store for live/paper trading sessions.

Shares the same SQLite database as ExperimentStore (~/.tensortrade/experiments.db).
Tables are created by ExperimentStore._create_tables() on startup.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from tensortrade.training.experiment_store import DEFAULT_DB_PATH


@dataclass
class LiveSession:
    id: str
    experiment_id: str
    config: dict
    status: str  # running, stopped, error
    started_at: str
    stopped_at: str | None
    symbol: str
    timeframe: str
    initial_equity: float | None
    final_equity: float | None
    total_trades: int
    total_bars: int
    pnl: float
    max_drawdown_pct: float
    model_version: int


@dataclass
class LiveTrade:
    id: int | None
    session_id: str
    timestamp: str
    step: int
    side: str  # buy, sell
    symbol: str
    price: float
    size: float
    commission: float
    alpaca_order_id: str | None
    model_version: int


@dataclass
class LiveExperience:
    id: int | None
    session_id: str
    step: int
    timestamp: str
    observation: bytes
    action: int
    reward: float
    next_observation: bytes
    done: bool
    symbol: str
    price: float


class LiveTradingStore:
    """SQLite-backed persistent store for live/paper trading sessions."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    # --- Session CRUD ---

    def create_session(
        self,
        experiment_id: str,
        symbol: str,
        timeframe: str,
        config: dict | None = None,
        initial_equity: float | None = None,
        model_version: int = 1,
    ) -> str:
        """Create a new live trading session and return its ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO live_sessions
               (id, experiment_id, config, status, started_at, symbol,
                timeframe, initial_equity, model_version)
               VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
            (
                session_id,
                experiment_id,
                json.dumps(config or {}),
                now,
                symbol,
                timeframe,
                initial_equity,
                model_version,
            ),
        )
        self._conn.commit()
        return session_id

    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        final_equity: float | None = None,
        total_trades: int | None = None,
        total_bars: int | None = None,
        pnl: float | None = None,
        max_drawdown_pct: float | None = None,
    ) -> None:
        """Update fields on a live session. Only non-None values are written."""
        updates: list[str] = []
        params: list[str | float | int] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "stopped":
                updates.append("stopped_at = ?")
                params.append(datetime.now(UTC).isoformat())
        if final_equity is not None:
            updates.append("final_equity = ?")
            params.append(final_equity)
        if total_trades is not None:
            updates.append("total_trades = ?")
            params.append(total_trades)
        if total_bars is not None:
            updates.append("total_bars = ?")
            params.append(total_bars)
        if pnl is not None:
            updates.append("pnl = ?")
            params.append(pnl)
        if max_drawdown_pct is not None:
            updates.append("max_drawdown_pct = ?")
            params.append(max_drawdown_pct)
        if not updates:
            return
        params.append(session_id)
        self._conn.execute(
            f"UPDATE live_sessions SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> LiveSession | None:
        """Retrieve a single live session by ID."""
        row = self._conn.execute(
            "SELECT * FROM live_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(
        self,
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LiveSession]:
        """List live sessions with optional filters."""
        query = "SELECT * FROM live_sessions WHERE 1=1"
        params: list[str | int] = []
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_session(r) for r in rows]

    # --- Trade Logging ---

    def log_live_trade(
        self,
        session_id: str,
        step: int,
        side: str,
        symbol: str,
        price: float,
        size: float,
        commission: float = 0.0,
        alpaca_order_id: str | None = None,
        model_version: int = 1,
    ) -> int:
        """Log a single live trade. Returns the trade row ID."""
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO live_trades
               (session_id, timestamp, step, side, symbol, price, size,
                commission, alpaca_order_id, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                step,
                side,
                symbol,
                price,
                size,
                commission,
                alpaca_order_id,
                model_version,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_session_trades(
        self,
        session_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[LiveTrade]:
        """Get all trades for a live session."""
        rows = self._conn.execute(
            """SELECT * FROM live_trades WHERE session_id = ?
               ORDER BY step LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    # --- Experience Logging ---

    def log_experience(
        self,
        session_id: str,
        step: int,
        observation: bytes,
        action: int,
        reward: float,
        next_observation: bytes,
        done: bool,
        symbol: str,
        price: float,
    ) -> int:
        """Log a single experience tuple. Returns the row ID."""
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO live_experiences
               (session_id, step, timestamp, observation, action, reward,
                next_observation, done, symbol, price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                step,
                now,
                observation,
                action,
                reward,
                next_observation,
                int(done),
                symbol,
                price,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_recent_experiences(
        self,
        session_id: str,
        limit: int = 256,
    ) -> list[LiveExperience]:
        """Get the most recent experiences for a session (newest first)."""
        rows = self._conn.execute(
            """SELECT * FROM live_experiences WHERE session_id = ?
               ORDER BY step DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    # --- Helpers ---

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> LiveSession:
        return LiveSession(
            id=row["id"],
            experiment_id=row["experiment_id"],
            config=json.loads(row["config"]),
            status=row["status"],
            started_at=row["started_at"],
            stopped_at=row["stopped_at"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            initial_equity=row["initial_equity"],
            final_equity=row["final_equity"],
            total_trades=row["total_trades"],
            total_bars=row["total_bars"],
            pnl=row["pnl"],
            max_drawdown_pct=row["max_drawdown_pct"],
            model_version=row["model_version"],
        )

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> LiveTrade:
        return LiveTrade(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            step=row["step"],
            side=row["side"],
            symbol=row["symbol"],
            price=row["price"],
            size=row["size"],
            commission=row["commission"],
            alpaca_order_id=row["alpaca_order_id"],
            model_version=row["model_version"],
        )

    @staticmethod
    def _row_to_experience(row: sqlite3.Row) -> LiveExperience:
        return LiveExperience(
            id=row["id"],
            session_id=row["session_id"],
            step=row["step"],
            timestamp=row["timestamp"],
            observation=row["observation"],
            action=row["action"],
            reward=row["reward"],
            next_observation=row["next_observation"],
            done=bool(row["done"]),
            symbol=row["symbol"],
            price=row["price"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
