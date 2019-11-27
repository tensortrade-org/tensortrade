registry = {}


class Instrument:

    def __init__(self,
                 symbol: str,
                 precision: int,
                 name: str):
        self.symbol = symbol
        self.precision = precision
        self.name = name

        registry[symbol] = self


# Cryptocurrencies
BTC = Instrument('BTC', 8, 'bitcoin')
ETH = Instrument('ETH', 8, 'ethereum')
BCH = Instrument('BTH', 8, 'bitcoin cash')
LTC = Instrument('LTC', 2, 'litecoin')
XRP = Instrument('XRP', 2, 'XRP')


# Fiat Currency
USD = Instrument('USD', 2, 'U.S. dollar')
EUR = Instrument('EUR', 2, 'euro')


# Hard Currency
GOLD = Instrument('XAU', 1, 'Gold')