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
import tensortrade.slippage as slippage

from gym.spaces import Space, Box
from typing import List, Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline


class SimulatedExchange(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environments.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(
            dtype=self.default('dtype', np.float32),
            feature_pipeline=self.default('feature_pipeline', None),
            **kwargs
        )

        self._commission_percent = self.default('commission_percent', 0.3, kwargs)
        self._base_precision = self.default('base_precision', 2, kwargs)
        self._instrument_precision = self.default('instrument_precision', 8, kwargs)
        self._initial_balance = self.default('initial_balance', 1e4, kwargs)
        self._price_column = self.default('price_column', 'close', kwargs)
        self._pretransform = self.default('pretransform', True, kwargs)

        self.data_frame = self.default('data_frame', data_frame)

        model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(model) if isinstance(model, str) else model()

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return getattr(self, '_data_frame', None)

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self._data_frame = data_frame
            self._price_history = None
            return

        self._data_frame = data_frame
        self._price_history = data_frame[self._price_column]
        self._pre_transformed_columns = data_frame.columns

        if self._pretransform:
            self.transform_data_frame()

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline=FeaturePipeline):
        self._feature_pipeline = feature_pipeline

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

        return self._feature_pipeline

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
    def observation_columns(self) -> List[str]:
        if self._data_frame is None:
            return None

        data_frame = self._data_frame.iloc[0:10]

        if self._feature_pipeline is not None:
            data_frame = self._feature_pipeline.transform(data_frame)

        return data_frame.select_dtypes(include=[np.float, np.number]).columns

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def _next_observation(self) -> pd.DataFrame:
        lower_range = max((self._current_step - self._window_size, 0))
        upper_range = min(self._current_step + 1, len(self._data_frame))

        obs = self._data_frame.iloc[lower_range:upper_range]

        if not self._pretransform and self._feature_pipeline is not None:
            obs = self._feature_pipeline.transform(obs)

        if len(obs) < self._window_size:
            padding = np.zeros((self._window_size - len(obs), len(self.observation_columns)))
            padding = pd.DataFrame(padding, columns=self.observation_columns)
            obs = pd.concat([padding, obs], ignore_index=True)

        obs = obs.select_dtypes(include='number')

        self._current_step += 1

        return obs

    def transform_data_frame(self) -> bool:
        if self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(self._data_frame)

    def current_price(self, symbol: str) -> float:
        if self._price_history is not None:
            return float(self._price_history.iloc[self._current_step])

        return 0

    def _is_valid_trade(self, trade: Trade) -> bool:
        if trade.is_buy and self._balance < trade.amount * trade.price:
            return False
        elif trade.is_sell and self._portfolio.get(trade.symbol, 0) < trade.amount:
            return False

        return trade.amount >= self._min_trade_amount and trade.amount <= self._max_trade_amount

    def _update_account(self, trade: Trade):
        if self._is_valid_trade(trade) and not trade.is_hold:
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

            return filled_trade

        if filled_trade.is_buy:
            price_adjustment = (1 + commission)
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

        self._current_step = 0
        self._balance = self.initial_balance
        self._portfolio = {self.base_instrument: self.balance}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])
