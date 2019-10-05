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
from gym.spaces import Box

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class ContinuousActionStrategy(ActionStrategy):
    """Simple continuous strategy, which calculates the trade amount as a fraction of the total balance."""

    def __init__(self, instrument_symbol: str = 'BTC', max_allowed_slippage_percent: float = 1.0, dtype: DTypeString = np.float16):
        """
        Arguments:
            instrument_symbol: The exchange symbol of the instrument being traded. Defaults to 'BTC'.
            max_allowed_slippage: The maximum amount above the current price the strategy will pay for an instrument. Defaults to 1.0 (i.e. 1%).
            dtype: A type or str corresponding to the dtype of the `action_space`. Defaults to `np.float16`.
        """

        super().__init__(action_space=Box(0, 1, shape=(1, 1), dtype=dtype), dtype=dtype)

        self.instrument_symbol = instrument_symbol
        self.max_allowed_slippage_percent = max_allowed_slippage_percent

    def get_trade(self, action: TradeActionUnion) -> Trade:
        action_type, trade_amount = action
        trade_type = TradeType(int(action_type * len(TradeType)))

        current_price = self._exchange.current_price(symbol=self.instrument_symbol)
        base_precision = self._exchange.base_precision
        instrument_precision = self._exchange.instrument_precision

        amount = self._exchange.instrument_balance(self.instrument_symbol)
        price = current_price

        if trade_type is TradeType.MARKET_BUY or trade_type is TradeType.LIMIT_BUY:
            price_adjustment = 1 + (self.max_allowed_slippage_percent / 100)
            price = max(round(current_price * price_adjustment, base_precision), base_precision)
            amount = round(self._exchange.balance * 0.99 *
                           trade_amount / price, instrument_precision)

        elif trade_type is TradeType.MARKET_SELL or trade_type is TradeType.LIMIT_SELL:
            price_adjustment = 1 - (self.max_allowed_slippage_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_held = self._exchange.portfolio.get(self.instrument_symbol, 0)
            amount = round(amount_held * trade_amount, instrument_precision)

        return Trade(self.instrument_symbol, trade_type, amount, price)
