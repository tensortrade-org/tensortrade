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

from stochastic.processes.noise import GaussianNoise

from tensortrade.stochastic.processes.brownian_motion import brownian_motion_log_returns
from tensortrade.stochastic.utils.helpers import get_delta, scale_times_to_generate, convert_to_prices
from tensortrade.stochastic.utils.parameters import ModelParameters, default


def geometric_brownian_motion_log_returns(params: 'ModelParameters') -> 'np.array':
    """Constructs a sequence of log returns.

    When log returns are exponentiated, it produces a random Geometric
    Brownian Motion (GBM). The GBM is the stochastic process underlying
    the Black-Scholes options pricing formula.

    Parameters
    ----------
    params : `ModelParameters`
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        The log returns of a geometric brownian motion process
    """
    wiener_process = np.array(brownian_motion_log_returns(params))
    sigma_pow_mu_delta = (params.gbm_mu - 0.5 * pow(params.all_sigma, 2)) * params.all_delta
    return wiener_process + sigma_pow_mu_delta


def geometric_brownian_motion_levels(params: 'ModelParameters'):
    """Constructs a sequence of price levels for an asset which evolves
    according to a geometric brownian motion process.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        The price levels for the asset
    """
    return convert_to_prices(params, geometric_brownian_motion_log_returns(params))


def gbm(base_price: int = 1,
        base_volume: int = 1,
        start_date: str = '2010-01-01',
        start_date_format: str = '%Y-%m-%d',
        times_to_generate: int = 1000,
        time_frame: str = '1h',
        params: 'ModelParameters' = None) -> 'pd.DataFrame':
    """Generates price data from a GBM process.

    Parameters
    ----------
    base_price : int, default 1
        The base price to use for price generation.
    base_volume : int, default 1
        The base volume to use for volume generation.
    start_date : str, default '2010-01-01'
        The start date of the generated data
    start_date_format : str, default '%Y-%m-%d'
        The format for the start date of the generated data.
    times_to_generate : int, default 1000
        The number of bars to make.
    time_frame : str, default '1h'
        The time frame.
    params : `ModelParameters`, optional
        The model parameters.

    Returns
    -------
    `pd.DataFrame`
        The generated data frame containing the OHLCV bars.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """

    delta = get_delta(time_frame)
    times_to_generate = scale_times_to_generate(times_to_generate, time_frame)

    params = params or default(base_price, times_to_generate, delta)

    prices = geometric_brownian_motion_levels(params)

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
