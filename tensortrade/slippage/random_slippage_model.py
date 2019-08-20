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
from tensortrade.trades import Trade, TradeType


class RandomSlippageModel(SlippageModel):
    "A uniform random slippage model."

    def __init__(self, max_price_slippage_percent: float = 3.0, max_amount_slippage_percent: float = 0.0):
        """
        Arguments:
            max_price_slippage_percent: The maximum random slippage to be applied to the fill price. Defaults to 3.0 (i.e. 3%).
            max_amount_slippage_percent: The maximum random slippage to be applied to the fill amount. Defaults to 0.
        """
        self.max_price_slippage_percent = max_price_slippage_percent
        self.max_amount_slippage_percent = max_amount_slippage_percent

    def fill_order(self, trade: Trade) -> Trade:
        fill_amount = trade.amount * (1 - np.random.uniform(0, self.max_amount_slippage_percent))
        fill_price = trade.price

        if trade.trade_type is TradeType.MARKET_BUY or trade.trade_type is TradeType.LIMIT_BUY:
            fill_price = trade.price * (1 + np.random.uniform(0, self.max_price_slippage_percent))
        elif trade.trade_type is TradeType.MARKET_SELL or trade.trade_type is TradeType.LIMIT_SELL:
            fill_price = trade.price * (1 - np.random.uniform(0, self.max_price_slippage_percent))

        return Trade(trade.symbol, trade.trade_type, amount=fill_amount, price=fill_price)
