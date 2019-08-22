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

from abc import abstractmethod
from gym.spaces import Space, Box
from typing import Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import AssetExchange
from tensortrade.slippage import RandomSlippageModel


class SimulatedExchange(AssetExchange):
    """An asset exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(dtype=kwargs.get('dtype', np.float16))

        if data_frame is not None:
            self._data_frame = data_frame.astype(self._dtype)

        self._commission_percent = kwargs.get('commission_percent', 0.3)
        self._base_precision = kwargs.get('base_precision', 2)
        self._asset_precision = kwargs.get('asset_precision', 8)
        self._initial_balance = kwargs.get('initial_balance', 1E5)
        self._min_order_amount = kwargs.get('min_order_amount', 1E-3)

        self._min_trade_price = kwargs.get('min_trade_price', 1E-6)
        self._max_trade_price = kwargs.get('max_trade_price', 1E6)
        self._min_trade_amount = kwargs.get('min_trade_amount', 1E-3)
        self._max_trade_amount = kwargs.get('max_trade_amount', 1E6)

        max_allowed_slippage_percent = kwargs.get('max_allowed_slippage_percent', 3.0)

        SlippageModelClass = kwargs.get('slippage_model', RandomSlippageModel)
        self._slippage_model = SlippageModelClass(max_allowed_slippage_percent)

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return self._data_frame

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        self._data_frame = data_frame

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
    def observation_space(self) -> Space:
        df_len = len(self._data_frame.ix[0])
        low = sum((self._min_trade_price,) * 4, (self._min_trade_amount, ))
        high = sum((self._max_trade_price,) * 4, (self._max_trade_amount, ))
        dtypes = (self._dtype, ) * df_len

        return Box(low=low, high=high, shape=(1, df_len), dtype=dtypes)

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame)

    def next_observation(self) -> pd.DataFrame:
        obs = self._data_frame.ix[self._current_step]

        self._current_step += 1

        return obs

    def current_price(self, symbol: str) -> float:
        if len(self._data_frame) is 0:
            self.next_observation()

        return float(self._data_frame['close'].values[self._current_step])

    def _is_valid_trade(self, trade: Trade) -> bool:
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

        is_trade_valid = self._is_valid_trade(trade)

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
