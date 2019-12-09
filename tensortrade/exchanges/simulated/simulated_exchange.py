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
from copy import deepcopy

from tensortrade.trades import Trade, TradeType, TradeSide
from tensortrade.orders import OrderStatus
from tensortrade.instruments import TradingPair, Quantity
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
        self._commission = self.default('commission_percent', 0.3, kwargs) / 100
        self._base_instrument = self.default('base_instrument', USD, kwargs)
        self._quote_instrument = self.default('quote_instrument', BTC, kwargs)
        self._initial_balance = self.default('initial_balance', 10000, kwargs)
        self._min_trade_size = self.default('min_trade_size', 1e-6, kwargs)
        self._max_trade_size = self.default('max_trade_size', 1e6, kwargs)
        self._min_trade_price = self.default('min_trade_price', 1e-8, kwargs)
        self._max_trade_price = self.default('max_trade_price', 1e8, kwargs)

        self._price_column = self.default('price_column', 'close', kwargs)
        self.data_frame = self.default('data_frame', data_frame)

        slippage_model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(slippage_model) if isinstance(
            slippage_model, str) else slippage_model()

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return getattr(self, '_data_frame', None)

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame = None):
        if not isinstance(data_frame, pd.DataFrame):
            self._data_frame = data_frame
            self._price_history = None
            return

        self._data_frame = data_frame
        self._pre_transformed_data = data_frame.copy()
        self._price_history = self._pre_transformed_data[self._price_column]
        self._pre_transformed_columns = self._pre_transformed_data.columns

    @property
    def observation_columns(self) -> List[str]:
        if self._data_frame is None:
            return None

        data_frame = self._data_frame.iloc[0:10]

        return data_frame.select_dtypes(include=[np.float, np.number]).columns

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def next_observation(self, window_size: int = 1) -> pd.DataFrame:
        lower_range = max(self._current_step - window_size, 0)
        upper_range = min(self._current_step + 1, len(self._data_frame))

        obs = self._data_frame.iloc[lower_range:upper_range]

        self._current_step += 1

        return obs

    def is_pair_tradeable(self, pair: TradingPair) -> bool:
        return pair.base == self._base_instrument and pair.quote == self._quote_instrument

    def quote_price(self, trading_pair: TradingPair) -> float:
        if self._price_history is not None:
            return float(self._price_history.iloc[self._current_step])

        return np.inf

    def _contain_price(self, price: float, precision: int) -> float:
        return round(max(min(price, self._max_trade_price), self._min_trade_price), precision)

    def _contain_size(self, size: float, precision: int) -> float:
        return round(max(min(size, self._max_trade_size), self._min_trade_size), precision)

    def _execute_buy_order(self, order: 'Order', base_wallet: 'Wallet', quote_wallet: 'Wallet', current_price: float) -> Trade:
        price_adjustment = (1 + self._commission)
        price = self._contain_price(current_price * price_adjustment, order.pair.base.precision)
        size = self._contain_size(order.size, order.pair.base.precision)

        if order.type == TradeType.MARKET:
            size = current_price * order.size / price
        elif order.price < current_price:
            return None

        size = min(size, base_wallet.balance.amount)

        trade = Trade(order.id, self.id, self._current_step,
                      order.pair, TradeSide.BUY, order.type, size, price)

        print('Before base wallet:', str(base_wallet))
        print('Before quote wallet:', str(quote_wallet))

        base_wallet -= Quantity(order.pair.base, trade.size, order.id)
        quote_wallet += Quantity(order.pair.quote, trade.size / trade.price, order.id)

        print('After base wallet:', str(base_wallet))
        print('After quote wallet:', str(quote_wallet))

        return trade

    def _execute_sell_order(self, order: 'Order', base_wallet: 'Wallet', quote_wallet: 'Wallet', current_price: float) -> Trade:
        price_adjustment = (1 - self._commission)
        price = self._contain_price(current_price * price_adjustment, order.pair.base.precision)
        size = self._contain_size(order.size, order.pair.base.precision)

        if order.type == TradeType.LIMIT and order.price > current_price:
            return None

        size = min(size / price, quote_wallet.balance.amount / price)

        trade = Trade(order.id, self.id, self._current_step,
                      order.pair, TradeSide.SELL, order.type, size, price)

        print('Before base wallet:', str(base_wallet))
        print('Before quote wallet:', str(quote_wallet))

        base_wallet += Quantity(order.pair.base, trade.size * trade.price, order.id)
        quote_wallet -= Quantity(order.pair.quote, trade.size, order.id)

        print('After base wallet:', str(base_wallet))
        print('After quote wallet:', str(quote_wallet))

        return trade

    def execute_order(self, order: 'Order', portfolio: 'Portfolio'):
        base_wallet = portfolio.get_wallet(self.id, order.pair.base)
        quote_wallet = portfolio.get_wallet(self.id, order.pair.quote)
        current_price = self.quote_price(order.pair)

        if order.is_buy:
            trade = self._execute_buy_order(order, base_wallet, quote_wallet, current_price)
        elif order.is_sell:
            trade = self._execute_sell_order(order, base_wallet, quote_wallet, current_price)

        if isinstance(trade, Trade):
            self._slippage_model.adjust_trade(trade)

            order.fill(self, trade)

        if order.status != OrderStatus.FILLED:
            order.cancel()

    def reset(self):
        self._current_step = 0
