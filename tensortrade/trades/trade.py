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


class Trade(object):
    """A trade object for use within trading environments."""

    def __init__(self, symbol: str, trade_type: 'TradeType', amount: float, price: float):
        """
        Arguments:
            symbol: The exchange symbol of the instrument in the trade (AAPL, ETH/USD, NQ1!, etc).
            trade_type: The type of trade executed (0 = HOLD, 1=LIMIT_BUY, 2=MARKET_BUY, 3=LIMIT_SELL, 4=MARKET_SELL).
            amount: The amount of the instrument in the trade (shares, satoshis, contracts, etc).
            price: The price paid per instrument in terms of the base instrument (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
        """
        self._symbol = symbol
        self._trade_type = trade_type
        self._amount = amount
        self._price = price

    def copy(self) -> 'Trade':
        """Return a copy of the current trade object."""
        return Trade(symbol=self._symbol, trade_type=self._trade_type, amount=self._amount, price=self._price)

    @property
    def symbol(self) -> str:
        """The exchange symbol of the instrument in the trade (AAPL, ETH/USD, NQ1!, etc)."""
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: str):
        self._symbol = symbol

    @property
    def trade_type(self) -> 'TradeType':
        """The type of trade ("buy", "sell", "hold", etc)."""
        return self._trade_type

    @trade_type.setter
    def trade_type(self, trade_type: 'TradeType'):
        self._trade_type = trade_type

    @property
    def amount(self) -> float:
        """The amount of the instrument in the trade (shares, satoshis, contracts, etc)."""
        return self._amount

    @amount.setter
    def amount(self, amount: float):
        self._amount = amount

    @property
    def price(self) -> float:
        """The price paid per instrument in terms of the base instrument (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD")."""
        return self._price

    @price.setter
    def price(self, price: float):
        self._price = price

    @property
    def is_hold(self) -> bool:
        """
        Returns:
            Whether the trade type is non-existent (i.e. hold).
        """
        return self._trade_type.is_hold

    @property
    def is_buy(self) -> bool:
        """
        Returns:
            Whether the trade type is a buy offer.
        """
        return self._trade_type.is_buy

    @property
    def is_sell(self) -> bool:
        """
        Returns:
            Whether the trade type is a sell offer.
        """
        return self._trade_type.is_sell
