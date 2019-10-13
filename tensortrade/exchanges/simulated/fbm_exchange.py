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
import numpy as np

from math import pi
from gym import spaces
from typing import Dict
from stochastic.continuous import FractionalBrownianMotion
from stochastic.noise import GaussianNoise

from tensortrade.trades import TradeType, Trade
from tensortrade.slippage import RandomUniformSlippageModel
from tensortrade.exchanges.simulated.simulated_exchange import SimulatedExchange


class FBMExchange(SimulatedExchange):
    """A simulated instrument exchange, in which the price history is based off a fractional brownian motion
    model with supplied parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(data_frame=None, **kwargs)

        self._base_price = kwargs.get('base_price', 10000)
        self._base_volume = kwargs.get('base_volume', 1)
        self._start_date = kwargs.get('start_date', '2010-01-01')
        self._start_date_format = kwargs.get('start_date_format', '%Y-%m-%d')
        self._times_to_generate = kwargs.get('times_to_generate', 100000)
        self._hurst = kwargs.get('hurst', 0.61)
        self._timeframe = kwargs.get('timeframe', '1h')

    def _generate_price_history(self):
        price_fbm = FractionalBrownianMotion(t=self._times_to_generate, hurst=self._hurst)
        volume_gen = GaussianNoise(t=self._times_to_generate)

        start_date = pd.to_datetime(self._start_date, format=self._start_date_format)

        price_volatility = price_fbm.sample(self._times_to_generate, zero=False)
        prices = price_volatility + self._base_price
        volume_volatility = volume_gen.sample(self._times_to_generate)
        volumes = volume_volatility * price_volatility + self._base_volume

        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame(
            [], columns=['date', 'volume'], dtype=float)

        price_frame['date'] = pd.date_range(
            start=start_date, periods=self._times_to_generate, freq="1min")
        price_frame['price'] = abs(prices)

        volume_frame['date'] = price_frame['date'].copy()
        volume_frame['volume'] = abs(volumes)

        price_frame.set_index('date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='m', origin=start_date)

        volume_frame.set_index('date')
        volume_frame.index = pd.to_datetime(volume_frame.index, unit='m', origin=start_date)

        data_frame = price_frame['price'].resample(self._timeframe).ohlc()
        data_frame['volume'] = volume_frame['volume'].resample(self._timeframe).sum()

        self.data_frame = data_frame.astype(self._dtype)

    def reset(self):
        super().reset()

        self._generate_price_history()
