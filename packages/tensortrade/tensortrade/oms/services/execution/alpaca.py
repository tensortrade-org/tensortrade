"""Alpaca Markets execution service.

Submits market orders to Alpaca via the ``alpaca-py`` SDK and returns
TensorTrade ``Trade`` objects matching the simulated execution contract.

Note: Alpaca paper trading API requires integer ``qty`` for crypto orders
and does not support the ``notional`` parameter.  For sells, we use the
close-position endpoint (``DELETE /v2/positions/{symbol}``) to liquidate
fractional positions that cannot be sold via a regular order.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from tensortrade.oms.orders import TradeSide, TradeType

logger = logging.getLogger(__name__)


@dataclass
class AlpacaFill:
    """Lightweight record returned after an Alpaca order fills."""

    order_id: str
    side: TradeSide
    filled_qty: float
    filled_avg_price: float
    commission: float
    filled_at: datetime


class AlpacaExecutionService:
    """Submit market orders to Alpaca and translate fills into TensorTrade trades.

    Designed to be called from the live-trading loop in the same way as the
    simulated execution functions (``execute_order``), but communicates with
    the Alpaca REST API instead of adjusting wallet balances.

    Parameters
    ----------
    api_key : str
        Alpaca API key.
    secret_key : str
        Alpaca secret key.
    paper : bool
        Use the paper-trading endpoint (default ``True``).
    """

    def __init__(self, api_key: str, secret_key: str, *, paper: bool = True) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for live execution. Install with: uv pip install alpaca-py"
            ) from exc

        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper
        logger.info(
            "AlpacaExecutionService initialised (paper=%s)",
            paper,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        symbol: str,
        side: TradeSide,
        qty: float = 0,
        *,
        client_order_id: str | None = None,
    ) -> AlpacaFill | None:
        """Submit a market order and block until it fills (or fails).

        For BUY orders, ``qty`` is rounded down to an integer (Alpaca paper
        trading requires integer qty for crypto).  For SELL orders, the
        close-position endpoint is used to liquidate the full position
        including any fractional remainder.

        Parameters
        ----------
        symbol : str
            Alpaca-style symbol, e.g. ``"BTC/USD"``.
        side : TradeSide
            ``TradeSide.BUY`` or ``TradeSide.SELL``.
        qty : float
            Quantity of the *base* asset to trade.
        client_order_id : str | None
            Optional idempotency key.

        Returns
        -------
        AlpacaFill | None
            Filled order details, or ``None`` if the order was rejected.
        """
        if side == TradeSide.SELL:
            return self._close_position(symbol)

        return self._submit_buy(symbol, qty, client_order_id)

    def _submit_buy(
        self,
        symbol: str,
        qty: float,
        client_order_id: str | None = None,
    ) -> AlpacaFill | None:
        """Submit a BUY market order with integer qty."""
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        int_qty = math.floor(qty)
        if int_qty < 1:
            logger.warning(
                "BUY qty %.6f rounds to 0 for %s — skipping order "
                "(Alpaca paper trading requires integer qty for crypto)",
                qty,
                symbol,
            )
            return None

        logger.info(
            "Submitting BUY market order: %s qty %d (requested %.6f)",
            symbol,
            int_qty,
            qty,
        )

        request = MarketOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            qty=int_qty,
            client_order_id=client_order_id or str(uuid.uuid4()),
        )

        try:
            order = self._client.submit_order(request)
        except Exception:
            logger.exception("BUY order submission failed for %s", symbol)
            return None

        filled_order = self._wait_for_fill(order.id)
        if filled_order is None:
            return None

        filled_qty = float(filled_order.filled_qty or 0)
        filled_avg = float(filled_order.filled_avg_price or 0)

        return AlpacaFill(
            order_id=str(filled_order.id),
            side=TradeSide.BUY,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg,
            commission=0.0,
            filled_at=filled_order.filled_at or datetime.now(UTC),
        )

    def _close_position(self, symbol: str) -> AlpacaFill | None:
        """Close a crypto position via DELETE /v2/positions/{symbol}.

        This endpoint handles fractional quantities that cannot be sold
        through a regular order on the Alpaca paper trading API.
        """
        # Alpaca wants the symbol without the slash for the positions endpoint
        api_symbol = symbol.replace("/", "")

        logger.info("Closing position via DELETE /v2/positions/%s", api_symbol)

        try:
            close_response = self._client.close_position(api_symbol)
        except Exception:
            logger.exception("Close position failed for %s", api_symbol)
            return None

        # close_position returns an Order object — wait for it to fill
        order_id = close_response.id
        filled_order = self._wait_for_fill(order_id)
        if filled_order is None:
            return None

        filled_qty = float(filled_order.filled_qty or 0)
        filled_avg = float(filled_order.filled_avg_price or 0)

        return AlpacaFill(
            order_id=str(filled_order.id),
            side=TradeSide.SELL,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg,
            commission=0.0,
            filled_at=filled_order.filled_at or datetime.now(UTC),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_fill(
        self,
        order_id: str,
        max_attempts: int = 30,
        poll_interval_sec: float = 0.5,
    ) -> object | None:
        """Poll Alpaca until the order reaches a terminal state.

        Returns the filled order object, or ``None`` on timeout / rejection.
        """
        import time

        for _ in range(max_attempts):
            try:
                order = self._client.get_order_by_id(order_id)
            except Exception:
                logger.exception("Failed to poll order %s", order_id)
                return None

            status = getattr(order.status, "value", str(order.status))
            if status == "filled":
                logger.info("Order %s filled", order_id)
                return order
            if status in ("canceled", "expired", "rejected"):
                logger.warning("Order %s ended with status %s", order_id, status)
                return None

            time.sleep(poll_interval_sec)

        logger.warning("Timed out waiting for order %s to fill", order_id)
        return None

    # ------------------------------------------------------------------
    # Convenience: simulated-service-compatible signature
    # ------------------------------------------------------------------

    def execute_order(
        self,
        order: object,
        base_wallet: object,
        quote_wallet: object,
        current_price: float,
        options: object,
        clock: object,
    ) -> object | None:
        """Adapter matching the ``simulated.execute_order`` signature.

        This is intentionally loose-typed (``object``) so callers can pass
        the standard TensorTrade order/wallet/clock objects without the live
        module depending on every internal type.  The *real* execution goes
        through :meth:`execute`.

        Returns a dict describing the fill (not a full ``Trade``), since
        constructing a ``Trade`` requires wallet transfers that do not apply
        in live execution.
        """
        side = TradeSide.BUY if getattr(order, "is_buy", False) else TradeSide.SELL
        qty = float(getattr(getattr(order, "remaining", None), "size", 0))

        pair = getattr(order, "exchange_pair", None)
        symbol = str(pair) if pair else "BTC/USD"

        fill = self.execute(symbol=symbol, side=side, qty=qty)
        if fill is None:
            return None

        return {
            "order_id": fill.order_id,
            "step": getattr(clock, "step", 0),
            "side": fill.side,
            "trade_type": TradeType.MARKET,
            "quantity": fill.filled_qty,
            "price": fill.filled_avg_price,
            "commission": fill.commission,
            "filled_at": fill.filled_at,
        }
