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
from tensortrade.exchanges import AssetExchange


class SimpleDiscreteStrategy(ActionStrategy):
    """Simple discrete strategy, which calculates the trade amount as a fraction of the total balance."""

    def __init__(self, n_bins: int = 24, asset_symbol: str = 'BTC', max_allowed_slippage_percent: float = 3.0):
        """
        Args:
            n_bins: The number of bins to divide the total balance by. Defaults to 24 (i.e. 1/24, 2/24, ..., 24/24).
            asset_symbol: The exchange symbol of the asset being traded. Defaults to 'BTC'.
            max_allowed_slippage: The maximum amount above the current price the strategy will pay for an asset. Defaults to 3.0 (i.e. 3%).
        """
        super().__init__(action_space=Discrete(n_bins), dtype=np.int64)

        self.n_bins = n_bins
        self.asset_symbol = asset_symbol
        self.max_allowed_slippage_percent = max_allowed_slippage_percent

    @property
    def dtype(self):
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `SimpleDiscreteStrategy` due to the requirements of `gym.spaces.Discrete` spaces. ')

    def get_trade(self, action: TradeActionUnion) -> Trade:
        trade_type = TradeType(action % len(TradeType))
        trade_amount = float(1 / (action % self.n_bins + 1))

        current_price = self._exchange.current_price(symbol=self.asset_symbol)
        base_precision = self._exchange.base_precision
        asset_precision = self._exchange.asset_precision

        amount = 0
        price = current_price

        if trade_type is TradeType.MARKET_BUY or trade_type is TradeType.LIMIT_BUY:
            price_adjustment = 1 + (self.max_allowed_slippage_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount = round(self._exchange.balance * trade_amount / price, asset_precision)

        elif trade_type is TradeType.MARKET_SELL or trade_type is TradeType.LIMIT_SELL:
            price_adjustment = 1 - (self.max_allowed_slippage_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_held = self._exchange.portfolio.get(self.asset_symbol, 0)
            amount = round(amount_held * trade_amount, asset_precision)

        return Trade(self.asset_symbol, trade_type, amount, price)
