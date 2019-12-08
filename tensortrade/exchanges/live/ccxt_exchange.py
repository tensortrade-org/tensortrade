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

import time
import ccxt
import numpy as np
import pandas as pd

from typing import Dict, List, Union
from gym.spaces import Space, Box
from ccxt import BadRequest

from tensortrade.trades import Trade, TradeType, TradeSide
from tensortrade.instruments import TradingPair, BTC, ETH
from tensortrade.exchanges import Exchange

BTC_ETH_PAIR = TradingPair(BTC, ETH)


class CCXTExchange(Exchange):
    """An exchange for trading on CCXT-supported cryptocurrency exchanges."""

    def __init__(self, exchange: Union[ccxt.Exchange, str] = 'coinbase',  **kwargs):
        super(CCXTExchange, self).__init__(**kwargs)

        exchange = self.default('exchange', exchange)

        self._exchange = getattr(ccxt, exchange)() if \
            isinstance(exchange, str) else exchange

        self._credentials = self.default('credentials', None, kwargs)
        self._timeframe = self.default('timeframe', '10m', kwargs)
        self._observation_type = self.default('observation_type', 'ohlcv', kwargs)
        self._observation_pairs = self.default('observation_pairs', [BTC_ETH_PAIR], kwargs)
        self._async_timeout_in_ms = self.default('async_timeout_in_ms', 15, kwargs)
        self._max_trade_wait_in_sec = self.default('max_trade_wait_in_sec', 15, kwargs)
        self._exchange.enableRateLimit = self.default('enable_rate_limit', True, kwargs)

        self._markets = self._exchange.load_markets()
        self._observation_symbols = [self.pair_to_symbol(pair) for pair in self._observation_pairs]

    @property
    def data_frame(self) -> pd.DataFrame:
        return self._data_frame

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        self._data_frame = data_frame

        if len(self._data_frame) >= self._window_size:
            self._data_frame = self._data_frame.iloc[-(self._window_size - 1):]

    @property
    def trades(self) -> List[Trade]:
        trades = {}

        for key in self._markets.keys():
            trades[key] = self._exchange.fetch_my_trades()[key]

        return trades

    @property
    def pre_transformation_columns(self) -> List[str]:
        if self._observation_type == 'ohlcv':
            return np.array([['{}_open'.format(symbol),
                              '{}_high'.format(symbol),
                              '{}_low'.format(symbol),
                              '{}_close'.format(symbol),
                              '{}_volume'.format(symbol)] for symbol in self._observation_symbols]).flatten()

        return np.array([['{}_side'.format(symbol),
                          '{}_price'.format(symbol),
                          '{}_amount'.format(symbol),
                          '{}_cost'.format(symbol)] for symbol in self._observation_symbols]).flatten()

    @property
    def generated_columns(self) -> List[str]:
        if self.has_next_observation:
            self._generate_next_observation().columns

        return self.pre_transformation_columns

    @property
    def has_next_observation(self) -> bool:
        if self._observation_type == 'ohlcv':
            return self._exchange.has['fetchOHLCV']

        return self._exchange.has['fetchTrades']

    def _generate_next_observation(self) -> pd.DataFrame:
        observations = pd.DataFrame([], columns=self.pre_transformation_columns)

        for symbol in self._observation_symbols:
            if self._observation_type == 'ohlcv':
                ohlcv = self._exchange.fetch_ohlcv(symbol, self._timeframe)

                observations['{}_open'.format(symbol)] = [ohlcv[1]]
                observations['{}_high'.format(symbol)] = [ohlcv[2]]
                observations['{}_low'.format(symbol)] = [ohlcv[3]]
                observations['{}_close'.format(symbol)] = [ohlcv[4]]
                observations['{}_volume'.format(symbol)] = [ohlcv[5]]
            elif self._observation_type == 'trades':
                trades = self._exchange.fetch_trades(symbol)

                observations['{}_side'.format(symbol)] = [
                    0 if trade['side'] == 'buy' else 1 for trade in trades]
                observations['{}_price'.format(symbol)] = [trade['price'] for trade in trades]
                observations['{}_amount'.format(symbol)] = [trade['amount'] for trade in trades]
                observations['{}_cost'.format(symbol)] = [trade['cost'] for trade in trades]

        observations = pd.concat([self._data_frame, observations], ignore_index=True, sort=False)

        self.data_frame = observations

        if self._feature_pipeline is not None:
            obs = self._feature_pipeline.transform(obs)

        if len(obs) < self._window_size:
            padding = np.zeros((self._window_size - len(obs), len(self.observation_columns)))
            padding = pd.DataFrame(padding, columns=self.observation_columns)
            obs = pd.concat([padding, obs], ignore_index=True, sort=False)

        return obs

    def pair_to_symbol(self, pair: 'TradingPair') -> str:
        return '{}/{}'.format(pair.quote.symbol, pair.base.symbol)

    def quote_price(self, pair: 'TradingPair') -> float:
        symbol = self.pair_to_symbol(pair)

        try:
            return self._exchange.fetch_ticker(symbol)['close']
        except BadRequest:
            return np.inf

    def execute_order(self, order: 'Order'):
        if order.type == TradeType.LIMIT and order.side == TradeSide.BUY:
            executed_order = self._exchange.create_limit_buy_order(
                order.symbol, order.amount, order.price)
        elif order.type == TradeType.MARKET and order.side == TradeSide.BUY:
            executed_order = self._exchange.create_market_buy_order(order.symbol, order.amount)
        elif order.type == TradeType.LIMIT and order.side == TradeSide.SELL:
            executed_order = self._exchange.create_limit_sell_order(
                order.symbol, order.amount, order.price)
        elif order.type == TradeType.MARKET and order.side == TradeSide.SELL:
            executed_order = self._exchange.create_market_sell_order(order.symbol, order.amount)
        else:
            return order.copy()

        max_wait_time = time.time() + self._max_trade_wait_in_sec

        while order['status'] == 'open' and time.time() < max_wait_time:
            executed_order = self._exchange.fetch_order(order.id)

        if order['status'] == 'open':
            self._exchange.cancel_order(order.id)

        trade = Trade(order_id=order.id,
                      exchange_id=self.id,
                      step=order.step,
                      pair=order.pair,
                      side=order.side,
                      trade_type=order.type,
                      size=executed_order['filled'],
                      price=executed_order['price'])

        self._trades += [trade]

    def reset(self):
        super().reset()

        self._markets = self._exchange.load_markets()
        self._data_frame = pd.DataFrame([], columns=self.generated_columns)
        self._trades = []
