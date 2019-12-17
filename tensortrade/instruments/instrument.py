
from typing import Union
from sympy import Symbol

from .quantity import Quantity
from .trading_pair import TradingPair

registry = {}


class Instrument:
    """A financial instrument for use in trading."""

    def __init__(self, symbol: Union[Symbol, str], precision: int, name: str = None):
        self._symbol = Symbol(symbol) if isinstance(symbol, str) else symbol
        self._precision = precision
        self._name = name

        registry[symbol] = self

    @property
    def symbol(self) -> str:
        return str(self._symbol)

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: 'Instrument') -> bool:
        return self.symbol == other.symbol and self.precision == other.precision and self.name == other.name

    def __ne__(self, other: 'Instrument') -> bool:
        return self.symbol != other.symbol or self.precision != other.precision or self.name != other.name

    def __rmul__(self, other: float) -> Quantity:
        return Quantity(instrument=self, size=other)

    def __truediv__(self, other):
        if isinstance(other, Instrument):
            return TradingPair(self, other)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return str(self)


# Crypto-currency
BTC = Instrument('BTC', 8, 'Bitcoin')
ETH = Instrument('ETH', 8, 'Ethereum')
BCH = Instrument('BCH', 8, 'Bitcoin Cash')
LTC = Instrument('LTC', 2, 'Litecoin')
XRP = Instrument('XRP', 2, 'XRP')


# Fiat Currency
USD = Instrument('USD', 2, 'U.S. Dollar')
EUR = Instrument('EUR', 2, 'Euro')


# Hard Currency
GOLD = Instrument('XAU', 1, 'Gold')
