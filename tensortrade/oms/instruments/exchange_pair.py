
from decimal import Decimal


class ExchangePair:
    """A pair of financial instruments to be traded on a specific exchange.

    Parameters
    ----------
    exchange : `Exchange`
        An exchange that contains the `pair` for trading.
    pair : `TradingPair`
        A trading pair available on the `exchange`.
    """

    def __init__(self, exchange: "Exchange", pair: "TradingPair"):
        self.exchange = exchange
        self.pair = pair

    @property
    def price(self) -> "Decimal":
        """The quoted price of the trading pair. (`Decimal`, read-only)"""
        return self.exchange.quote_price(self.pair)

    @property
    def inverse_price(self) -> "Decimal":
        """The inverse price of the trading pair. (`Decimal, read-only)"""
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
        return "{}:{}".format(self.exchange.name, self.pair)

    def __repr__(self):
        return str(self)
