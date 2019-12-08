
from typing import Union
from sympy import Symbol

from .quantity import Quantity

registry = {}


class Instrument:
    """A financial instrument for use in trading."""

    def __init__(self, symbol: Union[Symbol, str], precision: int, name: str):
        self.symbol = Symbol(symbol) if isinstance(symbol, str) else symbol
        self.precision = precision
        self.name = name

        registry[symbol] = self

    def __rmul__(self, other: float) -> Quantity:
        return Quantity(instrument=self, amount=other)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return str(self)


# Cryptocurrencies
BTC = Instrument('BTC', 8, 'Bitcoin')
ETH = Instrument('ETH', 8, 'Ethereum')
BCH = Instrument('BTH', 8, 'Bitcoin Cash')
LTC = Instrument('LTC', 2, 'Litecoin')
XRP = Instrument('XRP', 2, 'XRP')


# Fiat Currency
USD = Instrument('USD', 2, 'U.S. Dollar')
EUR = Instrument('EUR', 2, 'Euro')


# Hard Currency
GOLD = Instrument('XAU', 1, 'Gold')
