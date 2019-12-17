
from tensortrade.base.exceptions import InvalidTradingPair


class TradingPair:
    """A pair of financial instruments to be traded on a specific exchange."""

    def __init__(self, base: 'Instrument', quote: 'Instrument'):

        if base == quote:
            raise InvalidTradingPair(base, quote)

        self._base = base
        self._quote = quote

    @property
    def base(self):
        return self._base

    @property
    def quote(self):
        return self._quote

    def __str__(self):
        return '{}/{}'.format(self.base.symbol, self.quote.symbol)

    def __repr__(self):
        return str(self)
