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

from tensortrade.trades import Trade, TradeType, TradeSide
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline
from tensortrade.instruments import USD, BTC


class SimulatedExchange(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environments.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(**kwargs)

        self._commission = self.default('commission_percent', 0.3, kwargs) / 100
        self._base_instrument = self.default('base_instrument', USD, kwargs)
        self._quote_instrument = self.default('quote_instrument', BTC, kwargs)
        self._initial_balance = self.default('initial_balance', 10000, kwargs)
        self._min_trade_amount = self.default('min_trade_amount', 1e-6, kwargs)
        self._max_trade_amount = self.default('max_trade_amount', 1e6, kwargs)
        self._min_trade_price = self.default('min_trade_price', 1e-8, kwargs)
        self._max_trade_price = self.default('max_trade_price', 1e8, kwargs)

        self._price_column = self.default('price_column', 'close', kwargs)
        self._pretransform = self.default('pretransform', True, kwargs)
        self.data_frame = self.default('data_frame', data_frame)

        slippage_model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(slippage_model) if isinstance(
            slippage_model, str) else slippage_model()

    def transform_data_frame(self) -> bool:
        if self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(self._pre_transformed_data)

    @property
    def window_size(self) -> int:
        """The window size of observations."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

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
        self._pre_transformed_data = data_frame.copy()
        self._price_history = self._pre_transformed_data[self._price_column]
        self._pre_transformed_columns = self._pre_transformed_data.columns

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
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def generated_columns(self) -> List[str]:
        if self._data_frame is None:
            return None

        data_frame = self._data_frame.iloc[0:10]

        if self._feature_pipeline is not None:
            data_frame = self._feature_pipeline.transform(data_frame)

        return data_frame.select_dtypes(include=[np.float, np.number]).columns

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def _generate_next_observation(self) -> pd.DataFrame:
        lower_range = max(self._current_step - self._window_size, 0)
        upper_range = min(self._current_step + 1, len(self._data_frame))

        obs = self._data_frame.iloc[lower_range:upper_range]

        if not self._pretransform and self._feature_pipeline is not None:
            obs = self._feature_pipeline.transform(obs)

        if len(obs) < self._window_size:
            padding = np.zeros((self._window_size - len(obs), len(self.observation_columns)))
            padding = pd.DataFrame(padding, columns=self.observation_columns)
            obs = pd.concat([padding, obs], ignore_index=True, sort=False)

        obs = obs.select_dtypes(include='number')

        self._current_step += 1

        return obs

    def quote_price(self, trading_pair: 'TradingPair') -> float:
        if self._price_history is not None:
            return float(self._price_history.iloc[self._current_step])

        return np.inf

    def _execute_buy_order(self, order: 'Order', base_wallet: 'Wallet', quote_wallet: 'Wallet', current_price: float):
        price_adjustment = (1 + self._commission)
        price = round(current_price * price_adjustment, order.pair.base.precision)
        size = round(order.price * order.size / order.price, order.pair.base.precision)

        trade = Trade(order.id, self.id, self._current_step,
                      order.pair, TradeSide.BUY, order.type, size, price)

        base_wallet -= trade.size
        quote_wallet += trade.size / trade.price

        return trade

    def _execute_sell_order(self, order: 'Order', base_wallet: 'Wallet', quote_wallet: 'Wallet', current_price: float):
        price_adjustment = (1 - self._commission)
        price = round(current_price * price_adjustment, order.pair.base.precision)
        size = round(order.size, order.pair.base.precision)

        trade = Trade(order.id, self.id, self._current_step,
                      order.pair, TradeSide.SELL, order.type, size, price)

        base_wallet += trade.size
        quote_wallet -= trade.size / trade.price

        return trade

    def execute_order(self, order: 'Order'):
        current_price = self.quote_price(order.pair)
        base_wallet = self._portfolio.get_wallet(str(self.id), order.pair.base)
        quote_wallet = self._portfolio.get_wallet(str(self.id), order.pair.quote)

        if order.is_buy:
            trade = self._execute_buy_order(order, base_wallet, quote_wallet, current_price)
        elif order.is_sell:
            trade = self._execute_sell_order(order, base_wallet, quote_wallet, current_price)

        if isinstance(trade, Trade):
            self._slippage_model.adjust_trade(trade)
            self._trades += [trade]

            order.fill(self, trade)

    def reset(self):
        super().reset()

        self._current_step = 0
        self._trades = []
