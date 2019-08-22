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
import numpy as np
import pandas as pd

from typing import Dict, List
from gym.spaces import Space, Box
from ccxt import Exchange

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import AssetExchange


class CCXTExchange(AssetExchange):
    """An asset exchange for trading on CCXT-supported cryptocurrency exchanges."""

    def __init__(self, exchange: Exchange, base_asset: str = 'BTC', **kwargs):
        super().__init__(dtype=kwargs.get('dtype', np.float16))

        self._exchange = exchange
        self._base_asset = base_asset

        self._exchange.enableRateLimit = kwargs.get('enable_rate_limit', True)

        self._observation_type = kwargs.get('observation_type', 'trades')
        self._observation_symbol = kwargs.get('observation_symbol', 'ETH/BTC')
        self._observation_timeframe = kwargs.get('observation_timeframe', '10m')
        self._observation_window_size = kwargs.get('observation_window_size', 10)

        self._async_timeout_in_ms = kwargs.get('async_timeout_in_ms', 15)
        self._max_trade_wait_in_sec = kwargs.get('max_trade_wait_in_sec', 60)

    @property
    def base_precision(self) -> float:
        return self._markets[self._observation_symbol]['precision']['base']

    @base_precision.setter
    def base_precision(self, base_precision: float):
        raise ValueError('Cannot set the precision of `ccxt` exchanges.')

    @property
    def asset_precision(self) -> float:
        return self._markets[self._observation_symbol]['precision']['quote']

    @asset_precision.setter
    def asset_precision(self, asset_precision: float):
        raise ValueError('Cannot set the precision of `ccxt` exchanges.')

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        return self._exchange.fetch_free_balance()[self._base_asset]

    @property
    def portfolio(self) -> Dict[str, float]:
        portfolio = self._exchange.fetch_free_balance()

        return {k: v for k, v in portfolio.items() if v > 0}

    @property
    def trades(self) -> List[Trade]:
        trades = {}

        for key in self._markets.keys():
            trades[key] = self._exchange.fetch_my_trades()[key]

        return trades

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def observation_space(self) -> Space:
        low_price = self._markets[self._observation_symbol]['limits']['price']['min']
        high_price = self._markets[self._observation_symbol]['limits']['price']['max']
        low_volume = self._markets[self._observation_symbol]['limits']['amount']['min']
        high_volume = self._markets[self._observation_symbol]['limits']['amount']['max']

        if self._observation_type == 'ohlcv':
            low = (low_price, low_price, low_price, low_price, low_volume)
            high = (high_price, high_price, high_price, high_price, high_volume)
            dtypes = (self._dtype, self._dtype, self._dtype, self._dtype, np.int64)
            obs_shape = (self._observation_window_size, 5)
        else:
            low = (0, low_price, low_price, low_price, low_volume)
            high = (1, high_price, high_volume, high_price * high_volume)
            dtypes = (np.int8, self._dtype, self._dtype, self._dtype)
            obs_shape = (self._observation_window_size, 4)

        return Box(low=low, high=high, shape=obs_shape, dtype=dtypes)

    @property
    def has_next_observation(self) -> bool:
        if self._observation_type == 'ohlcv':
            return self._exchange.has['fetchOHLCV']

        return self._exchange.has['fetchTrades']

    def next_observation(self) -> pd.DataFrame:
        if self._observation_type == 'ohlcv':
            ohlcv = self._exchange.fetch_ohlcv(
                self._observation_symbol, timeframe=self._observation_timeframe)

            obs = [l[1:] for l in ohlcv]
        elif self._observation_type == 'trades':
            trades = self._exchange.fetch_trades(self._observation_symbol)

            obs = [[0 if t['side'] == 'buy' else 1, t['price'], t['amount'], t['cost']]
                   for t in trades]

        if len(obs) < self._observation_window_size:
            return np.pad(obs, (self._observation_window_size - len(obs), len(obs[0])))

        return obs

    def current_price(self, symbol: str) -> float:
        return self._exchange.fetch_ticker(symbol)['close']

    def execute_trade(self, trade: Trade) -> Trade:
        if trade.trade_type == TradeType.LIMIT_BUY:
            order = self._exchange.create_limit_buy_order(
                symbol=trade.symbol, amount=trade.amount, price=trade.price)
        elif trade.trade_type == TradeType.MARKET_BUY:
            order = self._exchange.create_market_buy_order(symbol=trade.symbol, amount=trade.amount)
        elif trade.trade_type == TradeType.LIMIT_SELL:
            order = self._exchange.create_limit_sell_order(
                symbol=trade.symbol, amount=trade.amount, price=trade.price)
        elif trade.trade_type == TradeType.MARKET_SELL:
            order = self._exchange.create_market_sell_order(
                symbol=trade.symbol, amount=trade.amount)

        max_wait_time = time.time() + self._max_trade_wait_in_sec

        while order['status'] is 'open' and time.time() < max_wait_time:
            order = self._exchange.fetch_order(order.id)

        if order['status'] is 'open':
            self._exchange.cancel_order(order.id)

        self._performance.append({
            'balance': self.balance,
            'net_worth': self.net_worth,
        }, ignore_index=True)

        return Trade(symbol=trade.symbol, trade_type=trade.trade_type, amount=order['filled'], price=order['price'])

    def reset(self):
        self._markets = self._exchange.load_markets()
        self._initial_balance = self._exchange.fetch_free_balance()[self._base_asset]
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])
