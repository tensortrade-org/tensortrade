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

import re

from typing import Callable

import numpy as np
import pandas as pd

from stochastic.processes.noise import GaussianNoise

from tensortrade.stochastic.utils.parameters import ModelParameters, default


def scale_times_to_generate(times_to_generate: int, time_frame: str) -> int:
    """Adjusts the number of times to generate the prices based on a time frame.

    Parameters
    ----------
    times_to_generate : int
        The number of time to generate prices.
    time_frame : str
        The time frame for generating.
        (e.g. 1h, 1min, 1w, 1d)

    Returns
    -------
    int
        The adjusted number of times to generate.

    Raises
    ------
    ValueError
        Raised if the `time_frame` provided does not match the correct format.
    """

    if 'MIN' in time_frame.upper():
        times_to_generate *= int(re.findall(r'\d+', time_frame)[0])
    elif 'H' in time_frame.upper():
        times_to_generate *= int(re.findall(r'\d+', time_frame)[0]) * 60
    elif 'D' in time_frame.upper():
        times_to_generate *= int(re.findall(r'\d+', time_frame)[0]) * 60 * 24
    elif 'W' in time_frame.upper():
        times_to_generate *= int(re.findall(r'\d+', time_frame)[0]) * 60 * 24 * 7
    elif 'M' in time_frame.upper():
        times_to_generate *= int(re.findall(r'\d+', time_frame)[0]) * 60 * 24 * 7 * 30
    else:
        raise ValueError('Timeframe must be either in minutes (min), hours (H), days (D), weeks (W), or months (M)')

    return times_to_generate


def get_delta(time_frame: str) -> float:
    """Gets the time delta for a given time frame.

    Parameters
    ----------
    time_frame : str
        The time frame for generating.
        (e.g. 1h, 1min, 1w, 1d)

    Returns
    -------
    float
        The time delta for the given time frame.
    """
    if 'MIN' in time_frame.upper():
        return 1 / (252 * 24 * (60 / int(time_frame.split('MIN')[0])))
    elif 'H' in time_frame.upper():
        return 1 / (252 * (24 / int(time_frame.split('H')[0])))
    elif 'D' in time_frame.upper():
        return 1 / 252
    elif 'M' in time_frame.upper():
        return 1 / 12


def convert_to_prices(param: 'ModelParameters', log_returns: 'np.array') -> 'np.array':
    """Converts a sequence of log returns into normal returns (exponentiation)
    and then computes a price sequence given a starting price, param.all_s0.

    Parameters
    ----------
    param: `ModelParameters`
        The model parameters.
    log_returns : `np.array`
        The log returns.

    Returns
    -------
    `np.array`
        The price sequence.
    """
    returns = np.exp(log_returns)
    # A sequence of prices starting with param.all_s0
    price_sequence = [param.all_s0]
    for i in range(1, len(returns)):
        # Add the price at t-1 * return at t
        price_sequence += [price_sequence[i - 1] * returns[i - 1]]

    return np.array(price_sequence)


def generate(price_fn: 'Callable[[ModelParameters], np.array]',
             base_price: int = 1,
             base_volume: int = 1,
             start_date: str = '2010-01-01',
             start_date_format: str = '%Y-%m-%d',
             times_to_generate: int = 1000,
             time_frame: str = '1h',
             params: ModelParameters = None) -> 'pd.DataFrame':
    """Generates a data frame of OHLCV data based on the price model specified.

    Parameters
    ----------
    price_fn : `Callable[[ModelParameters], np.array]`
        The price function generate the prices based on the chosen model.
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
        The data frame containing the OHLCV bars.
    """

    delta = get_delta(time_frame)
    times_to_generate = scale_times_to_generate(times_to_generate, time_frame)

    params = params or default(base_price, times_to_generate, delta)

    prices = price_fn(params)

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
