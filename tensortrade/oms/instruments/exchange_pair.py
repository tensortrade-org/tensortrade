from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensortrade.oms.exchanges.exchange import Exchange
    from tensortrade.oms.instruments.trading_pair import TradingPair


class ExchangePair:
    """A pair of financial instruments to be traded on a specific exchange.

    Attributes
    ----------
    exchange : `Exchange`
        An exchange that contains the `pair` for trading.
    pair : `TradingPair`
        A trading pair available on the `exchange`.
    """

    def __init__(self, exchange: Exchange, pair: TradingPair):
        """Initialize the exchange pair.

        Parameters
        ----------
        exchange : `Exchange`
            An exchange that contains the `pair` for trading.
        pair : `TradingPair`
            A trading pair available on the `exchange`.
        """
        self.exchange = exchange
        self.pair = pair

    @property
    def price(self) -> Decimal:
        """Get the quoted price of the trading pair (`Decimal`, read-only)."""
        return self.exchange.quote_price(self.pair)

    @property
    def inverse_price(self) -> Decimal:
        """Get the inverse price of the trading pair (`Decimal, read-only)."""
        quantization = Decimal(10) ** -self.pair.quote.precision
        return Decimal(self.price ** Decimal(-1)).quantize(quantization)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, ExchangePair):
            if str(self) == str(other):
                return True
        return False

    def __str__(self):
        return f"{self.exchange.name}:{self.pair}"

    def __repr__(self):
        return str(self)
