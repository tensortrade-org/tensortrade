"""Configuration for live / paper trading sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class LiveTradingConfig:
    """Immutable configuration for a live trading session.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTC/USD"``.
    timeframe : str
        Bar timeframe, e.g. ``"1h"``, ``"5m"``.
    paper : bool
        If ``True`` use Alpaca paper-trading endpoint.
    checkpoint_path : str
        Path to a saved RLlib checkpoint directory.
    experiment_id : str
        Reference back to the originating training experiment.
    feature_specs : list[dict[str, str | int | float]]
        Feature definitions from the experiment config (name, params, etc.).
    window_size : int
        Number of historical bars the agent observes.
    max_position_size_usd : float
        Maximum notional position size in USD.
    max_drawdown_pct : float
        Hard stop: halt trading if drawdown exceeds this percentage.
    """

    symbol: str = "BTC/USD"
    timeframe: str = "1h"
    paper: bool = True
    checkpoint_path: str = ""
    experiment_id: str = ""
    feature_specs: list[dict[str, str | int | float]] = field(default_factory=list)
    window_size: int = 10
    max_position_size_usd: float = 10_000.0
    max_drawdown_pct: float = 20.0

    # -- Alpaca credentials (loaded from env if not set) ----------------------

    @property
    def api_key(self) -> str:
        return os.environ.get("ALPACA_API_KEY", "")

    @property
    def secret_key(self) -> str:
        return os.environ.get("ALPACA_SECRET_KEY", "")

    @property
    def is_paper(self) -> bool:
        """Respect explicit ``paper`` flag *or* ALPACA_PAPER env var."""
        if self.paper:
            return True
        return os.environ.get("ALPACA_PAPER", "true").lower() in ("1", "true", "yes")

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty == valid)."""
        errors: list[str] = []
        if not self.api_key:
            errors.append("ALPACA_API_KEY environment variable is not set")
        if not self.secret_key:
            errors.append("ALPACA_SECRET_KEY environment variable is not set")
        if not self.checkpoint_path:
            errors.append("checkpoint_path is required")
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 100:
            errors.append("max_drawdown_pct must be between 0 and 100")
        return errors
