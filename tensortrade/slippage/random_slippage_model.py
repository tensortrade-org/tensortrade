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

from tensortrade.slippage import SlippageModel
from tensortrade.trades import Trade, TradeType, TradeSide


class RandomUniformSlippageModel(SlippageModel):
    """A uniform random slippage model."""

    def __init__(self, max_price_slippage_percent: float = 3.0, max_size_slippage_percent: float = 0.0):
        """
        Arguments:
            max_price_slippage_percent: The maximum random slippage to be applied to the fill price. Defaults to 3.0 (i.e. 3%).
            max_size_slippage_percent: The maximum random slippage to be applied to the fill size. Defaults to 0.
        """
        self.max_price_slippage_percent = self.default('max_price_slippage_percent',
                                                       max_price_slippage_percent)
        self.max_size_slippage_percent = self.default('max_size_slippage_percent',
                                                      max_size_slippage_percent)

    def adjust_trade(self, trade: Trade) -> Trade:
        size_slippage = np.random.uniform(0, self.max_size_slippage_percent / 100)
        price_slippage = np.random.uniform(0, self.max_price_slippage_percent / 100)

        initial_price = trade.price

        trade.size = trade.size * (1 - size_slippage)

        if trade.type == TradeType.MARKET and trade.side == TradeSide.BUY:
            trade.price = max(initial_price * (1 + price_slippage), 1e-3)
        elif trade.type == TradeType.LIMIT and trade.side == TradeSide.BUY:
            trade.price = max(initial_price * (1 + price_slippage), 1e-3)

            if trade.price > initial_price:
                trade.size *= min(trade.price / initial_price, 1)

        elif trade.type == TradeType.MARKET and trade.side == TradeSide.SELL:
            trade.price = max(initial_price * (1 - price_slippage), 1e-3)
        elif trade.type == TradeType.LIMIT and trade.side == TradeSide.SELL:
            trade.price = max(initial_price * (1 - price_slippage), 1e-3)

            if trade.price < initial_price:
                trade.size *= min(trade.price / initial_price, 1)

        return trade
