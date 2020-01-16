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


import re
import pandas as pd
import numpy as np

from stochastic.noise import GaussianNoise
from stochastic.continuous import FractionalBrownianMotion
from tensortrade.exchanges.simulated.simulated_exchange import SimulatedExchange
from tensortrade.exchanges.simulated.stochastic.stoch_gen import *


class StochasticExchange(SimulatedExchange):
    """A simulated instrument exchange, in which the price history is based off a
    *   Geometric Brownian Motion
    *   The Merton Jump-Diffusion Model
    *   The Heston Stochastic Volatility Model
    *   Cox Ingersoll Ross (CIR)
    *   Ornstein Uhlebneck stochastic process
    """

    def __init__(self, **kwargs):

        self._base_price = self.default('base_price', 1, kwargs)
        self._base_volume = self.default('base_volume', 1, kwargs)
        self._dtype = self.default('dtype', np.float32, kwargs)
        self._start_date = self.default('start_date', '2010-01-01', kwargs)
        self._start_date_format = self.default('start_date_format', '%Y-%m-%d', kwargs)
        self._times_to_generate = self.default('times_to_generate', 1000, kwargs)
        self._model_type = self.default('model_type', "FBM", kwargs)
        self._param_type = self.default('param_type', "Default", kwargs)
        self._hurst = self.default('hurst', 0.61, kwargs)
        self._timeframe = self.default('timeframe', '1H', kwargs)
        self._delta = self.default('delta', self.get_delta(self._timeframe), kwargs)
        self._scale_times_to_generate()
        self._model_params = self.default('model_params',
                                          self.get_model_params(self._param_type, self._base_price,
                                                                self._times_to_generate,
                                                                self._delta), kwargs)

        data_frame = self._generate_price_history()

        super().__init__(data_frame=data_frame, **kwargs)

    def _scale_times_to_generate(self):
        if 'MIN' in self._timeframe.upper():
            self._times_to_generate = self._times_to_generate * \
                                      int(re.findall(r'\d+', self._timeframe)[0])
        elif 'H' in self._timeframe.upper():
            self._times_to_generate = self._times_to_generate * \
                                      int(re.findall(r'\d+', self._timeframe)[0]) * 60
        elif 'D' in self._timeframe.upper():
            self._times_to_generate = self._times_to_generate * \
                                      int(re.findall(r'\d+', self._timeframe)[0]) * 60 * 24
        elif 'W' in self._timeframe.upper():
            self._times_to_generate = self._times_to_generate * \
                                      int(re.findall(r'\d+', self._timeframe)[0]) * 60 * 24 * 7
        elif 'M' in self._timeframe.upper():
            self._times_to_generate = self._times_to_generate * \
                                      int(re.findall(r'\d+', self._timeframe)[0]) * 60 * 24 * 7 * 30
        else:
            raise ValueError(
                'Timeframe must be either in minutes (min), hours (H), days (D), weeks (W), or months (M)')

    @staticmethod
    def get_price_data(model_type, mp):
        if model_type.upper() == "GBM":
            m = geometric_brownian_motion_levels(mp)
        elif model_type.upper() == "HESTON":
            m = heston_model_levels(mp)[0]
        elif model_type.upper() == "MERTON":
            m = geometric_brownian_motion_jump_diffusion_levels(mp)
        elif model_type.upper() == "COX":
            m = cox_ingersoll_ross_levels(mp)
        elif model_type.upper() == "ORNSTEIN UHLENBECK":
            m = ornstein_uhlenbeck_levels(mp)
        else:
            m = geometric_brownian_motion_levels(mp)

        return m

    @staticmethod
    def get_delta(time_frame):
        if 'MIN' in time_frame.upper():
            return 1 / (252 * 24 * (60/int(time_frame.split('MIN')[0])))
        elif 'H' in time_frame.upper():
            return 1 / (252 * (24/int(time_frame.split('H')[0])))
        elif 'D' in time_frame.upper():
            return 1 / 252
        elif 'M' in time_frame.upper():
            return 1 / 12

    def _generate_price_history(self):

        if self._model_type == 'FBM':
            price_fbm = FractionalBrownianMotion(t=self._times_to_generate, hurst=self._hurst)
            price_volatility = price_fbm.sample(self._times_to_generate, zero=False)
            prices = price_volatility + self._base_price

            volume_gen = GaussianNoise(t=self._times_to_generate)
            volume_volatility = volume_gen.sample(self._times_to_generate)
            volumes = volume_volatility * price_volatility + self._base_volume
        else:
            prices = self.get_price_data(self._model_type, self._model_params)
            volume_gen = GaussianNoise(t=self._times_to_generate)
            volumes = volume_gen.sample(self._times_to_generate)

        start_date = pd.to_datetime(self._start_date, format=self._start_date_format)
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

        return data_frame.astype(self._dtype)

    def reset(self):
        self.data_frame = self._generate_price_history()
