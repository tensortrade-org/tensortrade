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

    def __init__(self, instrument_symbol: str = 'BTC', position_size: int = 20, profit_target_range: range = range(0, 101, 5),
                 stop_loss_range: range = range(0, 101, 5), timeout_steps: int = np.iinfo(np.int32).max):
        """
        Arguments:
            instrument_symbol: The exchange symbol of the instrument being traded.
                Defaults to 'BTC'.
            position_size: The number of bins to divide the total balance by for each trade position.
                Defaults to 20 (i.e 1/20, 2/20 ... 20/20).
            profit_target_range: The range of percentages for the profit target of each trade position.
                Defaults to range(0, 100, 5).
            stop_loss_range: The range of percentages for the stop loss of each trade position.
                Defaults to range(0, 100, 5).
            timeout_steps (optional): Number of timesteps allowed per trade before automatically selling at market.
        """

        super().__init__(action_space=Discrete(position_size), dtype=np.int64)

        self.position_size = position_size
        self.instrument_symbol = instrument_symbol
        self.profit_target_range = profit_target_range
        self.stop_loss_range = stop_loss_range
        self.timeout_steps = timeout_steps
        self.trading_history = []
        self.current_step = 0

        self.reset()

    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `TargetStopActionStrategy` due to the requirements of `gym.spaces.Discrete` spaces. ')

    def reset(self):
        self.current_step = 0
        self.trading_history = list([])

    def get_trade(self, action: TradeActionUnion) -> Trade:
        """The trade type is determined by `action % len(TradeType)`, and the
        trade amount is determined by the multiplicity of the action.

        Using: 0 = HOLD, 1 = LIMIT_BUY|0.25, 2 = MARKET_BUY|0.25,
               5 = HOLD, 6 = LIMIT_BUY|0.5, 7 = MARKET_BUY|0.5, etc.

        For example, using this split of 4 sections, the input 7 (MARKET_BUY|0.5) / 5 (No. of possible actions)
        would return a remainder of 2, as the length of TradeType is equal to the amount of possible TradeTypes.
        The TradeType[2] is MARKET_BUY.
        Subsequently, 1 < (7 / 5) < 2 , therefore the trade amount would be calculated as:
        ((1) + 1) * ( 1 / 4 (No. of sections)) = 2 * 0.25 = 0.5
        ---------------------------------------------------------
        Profit target and stop loss percent are calculated as
        the inverse percentage of the trade amount of the maximum value in the range.

        For example, if the trade amount is 0.25, the profit target percent is:
        75% of the maximum value of self.profit_target_range
        (By default, 75% of 100; the profit target is then 0.75 * the traded price, or 75% above)

        The stop-loss in this case would be:
        -75% of the maximum value of self.stop_loss_range
        (By default, -75% of 100; the stop-loss is then -0.75 * the traded price, or 75% below.)
        """

        n_splits = self.position_size / len(TradeType)
        trade_type = TradeType(action % len(TradeType))

        current_price = self._exchange.current_price(symbol=self.instrument_symbol)
        base_precision = self._exchange.base_precision
        instrument_precision = self._exchange.instrument_precision

        amount = self._exchange.instrument_balance(self.instrument_symbol)
        current_price = round(current_price, base_precision)
        current_step = self.current_step

        for idx, trade in enumerate(self.trading_history):
            timeout_hit = current_step - trade[0] >= self.timeout_steps
            self.current_step += 1
            if trade[5] == TradeType.HOLD:
                pass
            elif timeout_hit:
                break
            else:
                profit_target_hit = current_price >= (trade[1] * trade[3])
                stop_loss_hit = current_price <= (trade[1] * trade[4])

                if profit_target_hit or stop_loss_hit:

                    if amount >= trade[2]:
                        amount = trade[2]
                    else:
                        amount = self._exchange.portfolio.get(self.instrument_symbol, 0)

                    del self.trading_history[idx]
                    self.current_step = 0
                    return Trade(self.instrument_symbol, TradeType.MARKET_SELL, amount, current_price)

        if action % len(TradeType) == 0:
            trade_amount = 0
        else:
            trade_amount = (int(action / len(TradeType)) + 1) * float(1 / n_splits)

        profit_target_percent = (max(self.profit_target_range) * (1 - trade_amount)) / 100

        stop_loss_percent = (-(max(self.profit_target_range) * (1 - trade_amount))) / 100

        if TradeType is TradeType.MARKET_BUY or TradeType is TradeType.LIMIT_BUY:
            amount = round(self._exchange.balance * 0.99 *
                           trade_amount / current_price, instrument_precision)

        elif TradeType is TradeType.MARKET_SELL or TradeType is TradeType.LIMIT_SELL:
            amount_held = self._exchange.portfolio.get(self.instrument_symbol, 0)
            amount = round(amount_held * trade_amount, instrument_precision)

        self.trading_history.append([current_step, current_price, amount,
                                     profit_target_percent, stop_loss_percent, trade_type])
        self.current_step = 0
        return Trade(self.instrument_symbol, trade_type, amount, current_price)
