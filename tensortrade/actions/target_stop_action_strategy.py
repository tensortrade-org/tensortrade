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
import numpy as np

from typing import Union
from gym.spaces import Discrete

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class TargetStopActionStrategy(ActionStrategy):

    def __init__(self, position_size: int = 20, instrument_symbol: str = 'BTC', profit_target: float = 1.0,
                 stop_loss: float = 1.0, trading_history: list = []):
        """
        Arguments:
            position_size: The number of bins to divide the total balance by. Defaults to 20 (i.e 1/20, 2/20 ... 20/20)
            instrument_symbol: The exchange symbol of the instrument being traded. Defaults to 'BTC'.
            profit_target: The amount of profit to be reached before trading the instrument. Defaults to 1.0 (i.e 1%)
            stop_loss: The amount of loss allowed before trading the instrument. Defaults to 1.0 (i.e 1%)
            trading_history: The history of trades recorded statefully in the program, denoted as
                             [instrument_symbol, price, amount]. Defaults to empty.

        """

        super().__init__(action_space=Discrete(position_size), dtype=np.int64)

        self.position_size = position_size
        self.instrument_symbol = instrument_symbol
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.trading_history = trading_history

    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `SimpleDiscreteStrategy` due to the requirements of `gym.spaces.Discrete` spaces. ')

    def get_trade(self, action: TradeActionUnion, test: bool = False, set_price: float = 1) -> Trade:
        """
        The trade type is determined by `action % len(TradeType)`,
        and the trade amount is determined by the multiplicity of the action.
        For example, 1 = LIMIT_BUY|0.25, 2 = MARKET_BUY|0.25, 6 = LIMIT_BUY|0.5, 7 = MARKET_BUY|0.5, etc.

        """
        n_splits = self.position_size / len(TradeType)
        trade_type = TradeType(action % len(TradeType))
        trade_amount = int(action / len(TradeType)) * float(1 / n_splits) + (1 / n_splits)

        if test is True:
            current_price = set_price
        else:
            current_price = self._exchange.current_price(symbol=self.instrument_symbol)
        base_precision = self._exchange.base_precision
        instrument_precision = self._exchange.instrument_precision

        amount = self._exchange.instrument_balance(self.instrument_symbol)
        price = current_price
        profit_percent = 1 + (self.profit_target / 100)
        stop_loss_percent = 1 - (self.stop_loss / 100)
        forced_trade = False
        trading_history_increment = 0

        for trade in self.trading_history:
            if self.instrument_symbol == trade[0] and forced_trade is not True:
                if price >= (trade[1] * profit_percent) or price <= (trade[1] * stop_loss_percent):
                    price = round(current_price, base_precision)
                    amount_held = self._exchange.portfolio.get(self.instrument_symbol, 0)
                    if amount_held >= trade[2]:
                        amount = trade[2]
                    else:
                        amount = round(amount_held * trade_amount, instrument_precision)
                    forced_trade = True
                    trade_type = TradeType.MARKET_SELL
                    del self.trading_history[trading_history_increment]
                    """
                    The program attempts to sell the amount that was bought, otherwise it defaults to the 
                    dynamically set trade amount. The forced_trade variable is set to True, the trade_type 
                    is set to sell, and the original trade is deleted from the history.
                    """
            trading_history_increment += 1

        if forced_trade is not True:
            if TradeType is TradeType.MARKET_BUY or TradeType is TradeType.LIMIT_BUY:
                price = round(current_price, base_precision)
                amount = round(self._exchange.balance * 0.99 *
                               trade_amount / price, instrument_precision)

            elif TradeType is TradeType.MARKET_SELL or TradeType is TradeType.LIMIT_SELL:
                price = round(current_price, base_precision)
                amount_held = self._exchange.portfolio.get(self.instrument_symbol, 0)
                amount = round(amount_held * trade_amount, instrument_precision)

        self.trading_history.append([self.instrument_symbol, price, amount])
        return Trade(self.instrument_symbol, trade_type, amount, price)
