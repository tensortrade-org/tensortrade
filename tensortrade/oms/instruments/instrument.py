# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


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

    def __hash__(self):
        return hash(self.symbol)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return str(self)


# Crypto
BTC = Instrument('BTC', 8, 'Bitcoin')
ETH = Instrument('ETH', 8, 'Ethereum')
XRP = Instrument('XRP', 8, 'XRP')
NEO = Instrument('NEO', 8, 'NEO')
BCH = Instrument('BCH', 8, 'Bitcoin Cash')
LTC = Instrument('LTC', 8, 'Litecoin')
ETC = Instrument('ETC', 8, 'Ethereum Classic')
XLM = Instrument('XLM', 8, 'Stellar Lumens')
LINK = Instrument('LINK', 8, 'Chainlink')
ATOM = Instrument('ATOM', 8, 'Cosmos')
DAI = Instrument('DAI', 8, 'Dai')
USDT = Instrument('USDT', 8, 'Tether')

# FX
USD = Instrument('USD', 2, 'U.S. Dollar')
EUR = Instrument('EUR', 2, 'Euro')
JPY = Instrument('JPY', 2, 'Japanese Yen')
KWN = Instrument('KWN', 2, 'Korean Won')
AUD = Instrument('AUD', 2, 'Australian Dollar')

# Commodities
XAU = Instrument('XAU', 2, 'Gold futures')
XAG = Instrument('XAG', 2, 'Silver futures')

# Stocks

AAPL = Instrument('AAPL', 2, 'Apple stock')
MSFT = Instrument('MSFT', 2, 'Microsoft stock')
TSLA = Instrument('TSLA', 2, 'Tesla stock')
AMZN = Instrument('AMZN', 2, 'Amazon stock')
