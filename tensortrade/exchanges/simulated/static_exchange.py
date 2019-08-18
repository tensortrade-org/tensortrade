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
import pandas as pd

from gym.spaces import Box
from typing import Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges.asset_exchange import AssetExchange
from tensortrade.slippage import RandomSlippageModel


class StaticExchange(AssetExchange):
    def __init__(self, data_frame: pd.DataFrame,  **kwargs):
        self._data_frame = data_frame

        self._commission_percent = kwargs.get('commission_percent', 0.3)
        self._base_precision = kwargs.get('base_precision', 2)
        self._asset_precision = kwargs.get('asset_precision', 8)
        self._initial_balance = kwargs.get('initial_balance', 1E5)
        self._max_allowed_slippage_percent = kwargs.get('max_allowed_slippage_percent', 3.0)
        self._min_order_amount = kwargs.get('min_order_amount', 1E-3)

        self._slippage_model = RandomSlippageModel(exchange=self,
                                                   max_price_slippage_percent=self._max_allowed_slippage_percent)

        self.reset()

    @property
    def base_precision(self):
        return self._base_precision

    @property
    def asset_precision(self):
        return self._asset_precision

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def observation_space(self):
        low_price, high_price, low_volume, high_volume = 1E-6, 1E6, 1E-3, 1E6

        low = (low_price, low_price, low_price, low_price, low_volume)
        high = (high_price, high_price, high_price, high_price, high_volume)
        dtypes = (self._dtype, self._dtype, self._dtype, self._dtype, np.int64)

        return Box(low=low, high=high, shape=(1, 5), dtype=dtypes)

    def current_price(self, symbol: str):
        if len(self._data_frame) is 0:
            self.next_observation()

        return float(self._data_frame['close'].values[self._current_step])

    def has_next_observation(self):
        return self._current_step < len(self._data_frame)

    def next_observation(self):
        self._current_step += 1

        return self._data_frame[self._current_step].values.astype(self._dtype)

    def is_valid_trade(self, trade: Trade) -> bool:
        if trade.trade_type is TradeType.MARKET_BUY or trade.trade_type is TradeType.LIMIT_BUY:
            return trade.amount >= self._min_order_amount and self._balance >= trade.amount * trade.price
        elif trade.trade_type is TradeType.MARKET_SELL or trade.trade_type is TradeType.LIMIT_SELL:
            return trade.amount >= self._min_order_amount and self._portfolio.get(trade.symbol, 0) >= trade.amount

        return True

    def _update_account(self, trade: Trade):
        if trade.amount > 0:
            self._trades = self._trades.append({
                'step': self._current_step,
                'symbol': trade.symbol,
                'type': trade.trade_type,
                'amount': trade.amount,
                'price': trade.price
            }, ignore_index=True)

        if trade.is_buy:
            self._balance -= trade.amount * trade.price
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.amount
        elif trade.is_sell:
            self._balance += trade.amount * trade.price
            self._portfolio[trade.symbol] -= trade.amount

        self._performance.append({
            'balance': self.balance,
            'net_worth': self.net_worth,
        }, ignore_index=True)

    def execute_trade(self, trade: Trade) -> Trade:
        current_price = self.current_price(symbol=trade.symbol)

        commission = self._commission_percent / 100

        is_trade_valid = self.is_valid_trade(trade)

        if trade.is_buy and is_trade_valid:
            price_adjustment = price_adjustment = (1 + commission)
            trade.price = round(current_price * price_adjustment, self._base_precision)
            trade.amount = round((trade.price * trade.amount) / trade.price, self._asset_precision)
        elif trade.is_sell and is_trade_valid:
            price_adjustment = (1 - commission)
            trade.price = round(current_price * price_adjustment, self._base_precision)
            trade.amount = round(trade.amount, self._asset_precision)

        filled_trade = self._slippage_model.fill_order(trade)

        self._update_account(filled_trade)

        return filled_trade

    def reset(self):
        self._balance = self._initial_balance

        self._portfolio = {}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])

        self._current_step = 0
