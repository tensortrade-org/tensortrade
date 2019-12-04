registry = {}


class Instrument:
    """A financial instrument to be traded on a specific exchange."""

    def __init__(self, symbol: str, exchange: str, precision: int, name: str):
        self.symbol = symbol
        self.exchange = exchange
        self.precision = precision
        self.name = name

        registry[symbol] = self


BINANCE_USDT = Instrument('USDT', 'Binance', 8, 'tether')
BINANCE_BTC = Instrument('BTC', 'Binance', 8, 'bitcoin')
BINANCE_ETH = Instrument('ETH', 'Binance', 8, 'ethereum')
BINANCE_BCH = Instrument('BTH', 'Binance', 8, 'bitcoin cash')
BINANCE_LTC = Instrument('LTC', 'Binance', 2, 'litecoin')
BINANCE_XRP = Instrument('XRP', 'Binance', 2, 'XRP')

COINBASE_PRO_USD = Instrument('BTC', 'Coinbase Pro', 8, 'U.S. dollar')
COINBASE_PRO_BTC = Instrument('BTC', 'Coinbase Pro', 8, 'bitcoin')
COINBASE_PRO_ETH = Instrument('ETH', 'Coinbase Pro', 8, 'ethereum')
COINBASE_PRO_BCH = Instrument('BTH', 'Coinbase Pro', 8, 'bitcoin cash')
COINBASE_PRO_LTC = Instrument('LTC', 'Coinbase Pro', 2, 'litecoin')
COINBASE_PRO_XRP = Instrument('XRP', 'Coinbase Pro', 2, 'XRP')
