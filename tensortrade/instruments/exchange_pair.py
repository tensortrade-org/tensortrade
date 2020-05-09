

from decimal import Decimal


class ExchangePair:
    """A pair of financial instruments to be traded on a specific exchange."""

    def __init__(self, exchange: 'Exchange', pair: 'TradingPair'):
        self._exchange = exchange
        self._pair = pair

    @property
    def exchange(self) -> 'Exchange':
        return self._exchange

    @property
    def pair(self) -> 'TradingPair':
        return self._pair

    @property
    def price(self) -> Decimal:
        return self.exchange.quote_price(self.pair)

    @property
    def inverse_price(self):
        price = self.exchange.quote_price(self.pair)
        quantization = Decimal(10) ** -self.pair.quote.precision
        return Decimal(price ** Decimal(-1)).quantize(quantization)

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
