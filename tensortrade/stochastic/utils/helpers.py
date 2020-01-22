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
import numpy as np
import pandas as pd

from typing import Callable
from stochastic.noise import GaussianNoise

from .parameters import ModelParameters, default


def scale_times_to_generate(times_to_generate: int, time_frame: str):

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


def get_delta(time_frame):
    if 'MIN' in time_frame.upper():
        return 1 / (252 * 24 * (60 / int(time_frame.split('MIN')[0])))
    elif 'H' in time_frame.upper():
        return 1 / (252 * (24 / int(time_frame.split('H')[0])))
    elif 'D' in time_frame.upper():
        return 1 / 252
    elif 'M' in time_frame.upper():
        return 1 / 12


def convert_to_returns(log_returns):
    """
    This method exponentiates a sequence of log returns to get daily returns.
    :param log_returns: the log returns to exponentiated
    :return: the exponentiated returns
    """
    return np.exp(log_returns)


def convert_to_prices(param, log_returns):
    """
    This method converts a sequence of log returns into normal returns (exponentiation) and then computes a price
    sequence given a starting price, param.all_s0.
    :param param: the model parameters object
    :param log_returns: the log returns to exponentiated
    :return:
    """
    returns = convert_to_returns(log_returns)
    # A sequence of prices starting with param.all_s0
    price_sequence = [param.all_s0]
    for i in range(1, len(returns)):
        # Add the price at t-1 * return at t
        price_sequence.append(price_sequence[i - 1] * returns[i - 1])
    return np.array(price_sequence)


def generate(price_fn: Callable[[ModelParameters], np.array],
             base_price: int = 1,
             base_volume: int = 1,
             start_date: str = '2010-01-01',
             start_date_format: str = '%Y-%m-%d',
             times_to_generate: int = 1000,
             time_frame: str = '1h',
             params: ModelParameters = None):

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
