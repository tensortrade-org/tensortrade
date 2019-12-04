class TradingPair():
    """A pair of financial instruments to be traded on a specific exchange."""

    def __init__(self, base_instrument: 'Instrument', quote_instrument: 'Instrument'):
        self.base_instrument = base_instrument
        self.quote_instrument = quote_instrument

    def __str__(self):
        return '{}/{}'.format(self.base_instrument.symbol, self.quote_instrument.symbol)

    def __repr__(self):
        return str(self)
