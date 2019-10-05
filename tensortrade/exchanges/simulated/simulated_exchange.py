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
from typing import Dict, Generator

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import InstrumentExchange
from tensortrade.slippage import RandomUniformSlippageModel


class SimulatedExchange(InstrumentExchange):
    """An instrument exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(base_instrument=kwargs.get('base_instrument', 'USD'),
                         dtype=kwargs.get('dtype', np.float16),
                         feature_pipeline=kwargs.get('feature_pipeline', None))

        self._should_pretransform_obs = kwargs.get('should_pretransform_obs', False)

        if data_frame is not None:
            self.data_frame = data_frame.astype(self._dtype)

        self._commission_percent = kwargs.get('commission_percent', 0.3)
        self._base_precision = kwargs.get('base_precision', 2)
        self._instrument_precision = kwargs.get('instrument_precision', 8)
        self._initial_balance = kwargs.get('initial_balance', 1E4)
        self._min_order_amount = kwargs.get('min_order_amount', 1E-3)
        self._window_size = kwargs.get('window_size', 1)

        self._min_trade_price = kwargs.get('min_trade_price', 1E-6)
        self._max_trade_price = kwargs.get('max_trade_price', 1E6)
        self._min_trade_amount = kwargs.get('min_trade_amount', 1E-3)
        self._max_trade_amount = kwargs.get('max_trade_amount', 1E6)

        max_allowed_slippage_percent = kwargs.get('max_allowed_slippage_percent', 1.0)

        SlippageModelClass = kwargs.get('slippage_model', RandomUniformSlippageModel)
        self._slippage_model = SlippageModelClass(max_allowed_slippage_percent)

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return self._data_frame

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        self._data_frame = data_frame

        if self._should_pretransform_obs and self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(self._data_frame)

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
    def generated_space(self) -> Space:
        low = np.array([self._min_trade_price, ] * 4 + [self._min_trade_amount, ])
        high = np.array([self._max_trade_price, ] * 4 + [self._max_trade_amount, ])

        return Box(low=low, high=high, dtype='float')

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def _create_observation_generator(self) -> Generator[pd.DataFrame, None, None]:
        for step in range(self._current_step, len(self._data_frame)):
            self._current_step = step

            obs = self._data_frame.iloc[step - self._window_size + 1:step + 1]

            if not self._should_pretransform_obs and self._feature_pipeline is not None:
                obs = self._feature_pipeline.transform(obs)

            yield obs

        raise StopIteration

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
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.amount

        self._portfolio[self._base_instrument] = self._balance

        self._performance = self._performance.append({
            'balance': self.balance,
            'net_worth': self.net_worth,
        }, ignore_index=True)

    def execute_trade(self, trade: Trade) -> Trade:
        current_price = self.current_price(symbol=trade.symbol)

        commission = self._commission_percent / 100

        filled_trade = trade.copy()

        if filled_trade.is_hold or not self._is_valid_trade(filled_trade):
            filled_trade.amount = 0
        elif filled_trade.is_buy:
            price_adjustment = price_adjustment = (1 + commission)
            filled_trade.price = max(round(current_price * price_adjustment,
                                           self._base_precision), self.base_precision)
            filled_trade.amount = round(
                (filled_trade.price * filled_trade.amount) / filled_trade.price, self._instrument_precision)
        elif filled_trade.is_sell:
            price_adjustment = (1 - commission)
            filled_trade.price = round(current_price * price_adjustment, self._base_precision)
            filled_trade.amount = round(filled_trade.amount, self._instrument_precision)

        filled_trade = self._slippage_model.fill_order(filled_trade, current_price)

        self._update_account(filled_trade)

        return filled_trade

    def reset(self):
        super().reset()

        self._balance = self._initial_balance

        self._portfolio = {self._base_instrument: self._balance}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])

        self._current_step = 0
