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

from typing import Any

from tensortrade.oms.instruments.quantity import Quantity
from tensortrade.oms.instruments.trading_pair import TradingPair


registry = {}


class Instrument:
    """A financial instrument for use in trading.

    Parameters
    ----------
    symbol : str
        The symbol used on an exchange for a particular instrument.
        (e.g. AAPL, BTC, TSLA)
    precision : int
        The precision the amount of the instrument is denoted with.
        (e.g. BTC=8, AAPL=1)
    name : str, optional
        The name of the instrument being created.
    """

    def __init__(self, symbol: str, precision: int, name: str = None) -> None:
        self.symbol = symbol
        self.precision = precision
        self.name = name

        registry[symbol] = self

    def __eq__(self, other: "Any") -> bool:
        """Checks if two instruments are equal.

        Parameters
        ----------
        other : `Any`
            The instrument being compared.

        Returns
        -------
        bool
            Whether the instruments are equal.
        """
        if not isinstance(other, Instrument):
            return False
        c1 = self.symbol == other.symbol
        c2 = self.precision == other.precision
        c3 = self.name == other.name
        return c1 and c2 and c3

    def __ne__(self, other: "Any") -> bool:
        """Checks if two instruments are not equal.

        Parameters
        ----------
        other : `Any`
            The instrument being compared.

        Returns
        -------
        bool
            Whether the instruments are not equal.
        """
        return not self.__eq__(other)

    def __rmul__(self, other: float) -> "Quantity":
        """Enables reverse multiplication.

        Parameters
        ----------
        other : float
            The number used to create a quantity.

        Returns
        -------
        `Quantity`
            The quantity created by the number and the instrument involved with
            this operation.
        """
        return Quantity(instrument=self, size=other)

    def __truediv__(self, other: "Instrument") -> "TradingPair":
        """Creates a trading pair through division.

        Parameters
        ----------
        other : `Instrument`
            The instrument that will be the quote of the pair.

        Returns
        -------
        `TradingPair`
            The trading pair created from the two instruments.

        Raises
        ------
        InvalidTradingPair
            Raised if `other` is the same instrument as `self`.
        Exception
            Raised if `other` is not an instrument.
        """
        if isinstance(other, Instrument):
            return TradingPair(self, other)
        raise Exception(f"Invalid trading pair: {other} is not a valid instrument.")

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
