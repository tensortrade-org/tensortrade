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

import pandas as pd
import numpy as np

from tensortrade.rewards import RewardStrategy
from tensortrade.trades import TradeType, Trade


class SimpleProfitStrategy(RewardStrategy):
    """A reward strategy that rewards the agent for profitable trades and prioritizes trading over not trading.

    This strategy supports simple action strategies that trade a single position in a single instrument at a time.
    """

    def reset(self):
        """Necessary to reset the last purchase price and state of open positions."""
        self._purchase_price = -1
        self._is_holding_instrument = False

    def get_reward(self, current_step: int, trade: Trade) -> float:
        """Reward -1 for not holding a position, 1 for holding a position, 2 for opening a position, and 1 + 5^(log_10(profit)) for closing a position.

        The 5^(log_10(profit)) function simply slows the growth of the reward as trades get large.
        """
        for t in trade:
            if t.is_hold and self._is_holding_instrument:
                yield 1
            elif t.is_buy and t.amount > 0:
                self._purchase_price = t.price
                self._is_holding_instrument = True

                yield 2
            elif t.is_sell and t.amount > 0:
                self._is_holding_instrument = False
                profit_per_instrument = t.price - self._purchase_price
                profit = t.amount * profit_per_instrument
                profit_sign = np.sign(profit)

                yield profit_sign * (1 + (5 ** np.log10(abs(profit))))

            yield -1
