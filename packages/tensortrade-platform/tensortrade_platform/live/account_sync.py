"""Account reconciliation with Alpaca Markets.

Queries the Alpaca trading API for account equity, buying power, and
current positions, then reconciles with local tracking state.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Snapshot of a single Alpaca position."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    side: str  # "long" or "short"


@dataclass
class AccountSnapshot:
    """Snapshot of the Alpaca account + positions at a point in time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    equity: float = 0.0
    buying_power: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    positions: list[PositionInfo] = field(default_factory=list)
    # Reconciliation
    local_position_qty: float = 0.0
    position_mismatch: bool = False


class AccountSync:
    """Query Alpaca for account state and reconcile with local tracking.

    Parameters
    ----------
    api_key : str
        Alpaca API key. Falls back to ``ALPACA_API_KEY`` env var.
    secret_key : str
        Alpaca secret key. Falls back to ``ALPACA_SECRET_KEY`` env var.
    paper : bool
        Use the paper-trading endpoint (default ``True``).
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        *,
        paper: bool = True,
    ) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for account sync. Install with: uv pip install alpaca-py"
            ) from exc

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._client = TradingClient(self._api_key, self._secret_key, paper=paper)
        self._paper = paper

        # Local position tracking (updated by the live trader)
        self._local_qty: float = 0.0
        self._local_symbol: str = ""
        self._peak_equity: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_local_position(self, symbol: str, qty: float) -> None:
        """Update the locally-tracked position (called after each trade)."""
        self._local_symbol = symbol
        self._local_qty = qty

    def snapshot(self) -> AccountSnapshot:
        """Fetch account + positions and reconcile with local state.

        Returns
        -------
        AccountSnapshot
            Current account state with reconciliation metadata.
        """
        snap = AccountSnapshot()

        try:
            account = self._client.get_account()
            snap.equity = float(account.equity or 0)
            snap.buying_power = float(account.buying_power or 0)
            snap.cash = float(account.cash or 0)
            snap.portfolio_value = float(getattr(account, "portfolio_value", 0) or 0)
        except Exception:
            logger.exception("Failed to fetch Alpaca account")
            return snap

        # Track peak equity for drawdown calculation
        if snap.equity > self._peak_equity:
            self._peak_equity = snap.equity

        # Fetch positions
        try:
            raw_positions = self._client.get_all_positions()
            for pos in raw_positions:
                snap.positions.append(
                    PositionInfo(
                        symbol=str(pos.symbol),
                        qty=float(pos.qty or 0),
                        avg_entry_price=float(pos.avg_entry_price or 0),
                        market_value=float(pos.market_value or 0),
                        unrealized_pl=float(pos.unrealized_pl or 0),
                        unrealized_plpc=float(pos.unrealized_plpc or 0),
                        current_price=float(pos.current_price or 0),
                        side=str(getattr(pos, "side", "long")),
                    )
                )
        except Exception:
            logger.exception("Failed to fetch Alpaca positions")

        # Reconcile
        snap.local_position_qty = self._local_qty
        remote_qty = self._get_position_qty(snap.positions, self._local_symbol)
        snap.position_mismatch = abs(remote_qty - self._local_qty) > 1e-8

        if snap.position_mismatch:
            logger.warning(
                "Position mismatch for %s: local=%.6f, remote=%.6f",
                self._local_symbol,
                self._local_qty,
                remote_qty,
            )

        return snap

    def compute_drawdown_pct(self) -> float:
        """Compute current drawdown percentage from peak equity.

        Returns
        -------
        float
            Drawdown as a positive percentage (0.0 means no drawdown).
        """
        if self._peak_equity <= 0:
            return 0.0
        try:
            account = self._client.get_account()
            current_equity = float(account.equity or 0)
        except Exception:
            logger.exception("Failed to fetch equity for drawdown calc")
            return 0.0

        if current_equity >= self._peak_equity:
            self._peak_equity = current_equity
            return 0.0

        return ((self._peak_equity - current_equity) / self._peak_equity) * 100.0

    def get_live_pnl(self) -> dict[str, float]:
        """Compute live PnL summary from Alpaca positions.

        Returns
        -------
        dict[str, float]
            Keys: ``unrealized_pnl``, ``unrealized_pnl_pct``,
            ``equity``, ``cash``, ``drawdown_pct``.
        """
        snap = self.snapshot()
        total_unrealized = sum(p.unrealized_pl for p in snap.positions)
        total_unrealized_pct = 0.0
        if snap.positions:
            total_cost = sum(p.avg_entry_price * p.qty for p in snap.positions)
            if total_cost > 0:
                total_unrealized_pct = (total_unrealized / total_cost) * 100.0

        return {
            "unrealized_pnl": total_unrealized,
            "unrealized_pnl_pct": total_unrealized_pct,
            "equity": snap.equity,
            "cash": snap.cash,
            "drawdown_pct": self.compute_drawdown_pct(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_position_qty(positions: list[PositionInfo], symbol: str) -> float:
        """Find the quantity for *symbol* in a list of positions."""
        if not symbol:
            return 0.0
        # Alpaca uses symbols like "BTCUSD" (no slash) for crypto
        normalised = symbol.replace("/", "")
        for pos in positions:
            if pos.symbol.replace("/", "") == normalised:
                return pos.qty
        return 0.0
