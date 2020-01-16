# Copyright 2020 The TensorTrade Authors.
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

from stochastic.noise import GaussianNoise

from .utils.brownian_motion import brownian_motion_log_returns
from .utils.helpers import get_delta, scale_times_to_generate
from .utils.parameters import ModelParameters, default


def cox_ingersoll_ross_levels(params):
    """
    Constructs the rate levels of a mean-reverting Cox-Ingersoll-Ross process.
    Used to model interest rates as well as stochastic volatility in the Heston
    model. We pass a correlated Brownian motion process into the method from
    which the interest rate levels are constructed because the returns between
    the underlying and the stochastic volatility should be correlated. The other
     correlated process is used in the Heston model.

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.
    Returns:
        The interest rate levels for the CIR process.
    """
    brownian_motion = brownian_motion_log_returns(params)
    # Setup the parameters for interest rates
    a, mu, zero = params.cir_a, params.cir_mu, params.all_r0
    # Assumes output is in levels
    levels = [zero]
    for i in range(1, params.all_time):
        drift = a * (mu - levels[i - 1]) * params.all_delta
        # The main difference between this and the Ornstein Uhlenbeck model is that we multiply the 'random'
        # component by the square-root of the previous level i.e. the process has level dependent interest rates.
        randomness = np.sqrt(levels[i - 1]) * brownian_motion[i - 1]
        levels.append(levels[i - 1] + drift + randomness)
    return np.array(levels)


def cox(base_price: int = 1,
        base_volume: int = 1,
        start_date: str = '2010-01-01',
        start_date_format: str = '%Y-%m-%d',
        times_to_generate: int = 1000,
        time_frame: str = '1h',
        model_params: ModelParameters = None):

    delta = get_delta(time_frame)
    times_to_generate = scale_times_to_generate(times_to_generate, time_frame)

    params = model_params or default(base_price, times_to_generate, delta)

    prices = cox_ingersoll_ross_levels(params)

    volume_gen = GaussianNoise(t=times_to_generate)
    volumes = volume_gen.sample(times_to_generate) + base_volume

    start_date = pd.to_datetime(start_date, format=start_date_format)
    price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
    volume_frame = pd.DataFrame([], columns=['date', 'volume'], dtype=float)

    price_frame['date'] = pd.date_range(start=start_date, periods=times_to_generate, freq="1min")
    price_frame['price'] = abs(prices)

    volume_frame['date'] = price_frame['date'].copy()
    volume_frame['volume'] = abs(volumes)

    price_frame.set_index('date')
    price_frame.index = pd.to_datetime(price_frame.index, unit='m', origin=start_date)

    volume_frame.set_index('date')
    volume_frame.index = pd.to_datetime(volume_frame.index, unit='m', origin=start_date)

    data_frame = price_frame['price'].resample(time_frame).ohlc()
    data_frame['volume'] = volume_frame['volume'].resample(time_frame).sum()

    return data_frame
