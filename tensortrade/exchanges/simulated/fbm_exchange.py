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

from gym import spaces
from typing import Dict

from tensortrade.trades import TradeType, Trade
from tensortrade.slippage import RandomSlippageModel
from tensortrade.exchanges import AssetExchange


class FBMExchange(AssetExchange):
    def __init__(self, **kwargs):
        self._commission_percent = kwargs.get('commission_percent', 0.3)
        self._base_precision = kwargs.get('base_precision', 2)
        self._asset_precision = kwargs.get('asset_precision', 8)
        self._initial_balance = kwargs.get('initial_balance', 1E5)
        self._max_allowed_slippage_percent = kwargs.get(
            'max_allowed_slippage_percent', 3.0)
        self._min_order_amount = kwargs.get('min_order_amount', 1E-5)

        self._base_price = kwargs.get('base_price', 3000)
        self._base_volume = kwargs.get('base_volume', 1000)
        self._hurst = kwargs.get('hurst', 0.58)

        # self._fbm = FractionalBrownianMotion(t=1, hurst=self._hurst)
        self._slippage_model = RandomSlippageModel(self._max_allowed_slippage_percent)

        self.reset()

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
        return spaces.Box(low=0, high=1, shape=(1, 5), dtype=self.dtype)

    @property
    def has_next_observation(self):
        return True

    def next_observation(self):
        dates = np.linspace(0, 1, 2)
        prices = []  # self._fbm.sample(1)
        volumes = []  # self._fbm.sample(1)

        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame(
            [], columns=['date', 'volume'], dtype=float)

        price_frame['date'] = pd.to_datetime(dates, unit="m")
        price_frame['price'] = prices

        volume_frame['date'] = pd.to_datetime(dates, unit="m")
        volume_frame['volume'] = volumes

        price_frame.set_index('date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='s')

        volume_frame.set_index('date')
        volume_frame.index = pd.to_datetime(price_frame.index, unit='s')

        ohlc = price_frame['price'].resample('1min').ohlc()
        volume = volume_frame['volume'].resample('1min').sum()

        ohlc.reset_index(drop=True)
        volume.reset_index(drop=True)

        self._data_frame = self._data_frame.append({
            'open': float(ohlc['open'].values[-1] * self._base_price),
            'high': float(ohlc['high'].values[-1] * self._base_price),
            'low': float(ohlc['low'].values[-1] * self._base_price),
            'close': float(ohlc['close'].values[-1] * self._base_price),
            'volume': float(volume.values[-1] * self._base_volume),
        }, ignore_index=True)

        self._current_step += 1

        return self._data_frame[-1].astype(self._dtype)

    def current_price(self, symbol: str, output_symbol: str = 'USD'):
        if len(self._data_frame) is 0:
            self.next_observation()

        return float(self._data_frame['close'].values[-1])

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
        self._trades = pd.DataFrame(
            [], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])

        self._data_frame = pd.DataFrame(
            [], columns=['open', 'high', 'low', 'close', 'volume'])
        self._data_frame.reset_index(drop=True)

        self._current_step = 1
