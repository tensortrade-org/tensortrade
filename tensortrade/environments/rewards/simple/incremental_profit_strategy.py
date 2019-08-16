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

from tensortrade.environments.rewards import RewardStrategy
from tensortrade.exchanges import AssetExchange


class IncrementalProfitStrategy(RewardStrategy):
    """Calculate the agents rewards based on the incremental profits made by its actions"""

    def __init__(self, base_symbol: str = 'USD', asset_symbol: str = 'BTC'):
        self.last_bought = 0
        self.last_sold_price = -1

        self.base_symbol = base_symbol
        self.asset_symbol = asset_symbol

    def get_reward(self, current_step: int, exchange: AssetExchange) -> float:
        reward = 0

        current_price = exchange.current_price(
            symbol=self.asset_symbol, output_symbol=self.base_symbol)
        performance = exchange.performance()
        curr_balance = exchange.balance(symbol=self.base_symbol)

        if len(performance) < 1:
            return reward

        prev_balance = performance['balance'].values[-1]

        if self.last_sold_price is -1:
            self.last_sold_price = current_price

        if curr_balance > prev_balance:
            reward = exchange.net_worth() - performance['net_worth'].values[self.last_bought]
            self.last_sold_price = current_price

        elif curr_balance < prev_balance:
            reward = self.last_sold_price - current_price
            self.last_bought = current_step

        return reward
