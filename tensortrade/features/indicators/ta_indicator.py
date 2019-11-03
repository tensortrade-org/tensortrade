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
# limitations under the License.

import ta
import numpy as np
import pandas as pd

from gym import Space
from copy import copy
from abc import abstractmethod
from typing import Union, List, Callable

from tensortrade.features.feature_transformer import FeatureTransformer


class TAIndicator(FeatureTransformer):
    """Adds one or more TA indicators to a data frame, based on existing open, high, low, close, and 'volume from'
     column values.."""

    # Indicators supported by TA module:
    indicators = [
        # Momentum
        ('AO', ta.ao, ['High', 'Close']),
        #('KAMA', ta.kama, ['Close']),
        ('MFI', ta.money_flow_index, ['High', 'Low', 'Close', 'VolumeFrom']),
        ('RSI', ta.rsi, ['Close']),
        ('SO', ta.stoch, ['Close']),
        ('SOS', ta.stoch_signal, ['High','Low', 'Close']),
        ('TSI', ta.tsi, ['Close']),
        ('UO', ta.uo, ['High', 'Low', 'Close']),
        ('WR', ta.wr, ['High', 'Low', 'Close']),
        # Volume
        ('ADI', ta.acc_dist_index, ['High', 'Low', 'Close', 'VolumeFrom']),
        ('CMF', ta.chaikin_money_flow, ['High', 'Low', 'Close', 'VolumeFrom']),
        ('EM', ta.ease_of_movement, ['High', 'Low', 'Close', 'VolumeFrom']),
        ('FI', ta.force_index, ['Close', 'Volume BTC']),
        ('NVI', ta.negative_volume_index, ['Close', 'Volume BTC']),
        ('OBV', ta.on_balance_volume, ['Close', 'Volume BTC']),
        ('PCR', ta.put_call_ratio, ['Close', 'Volume BTC']),
        ('VPT', ta.volume_price_trend, ['Close', 'Volume BTC']),
        # Volatility
        ('ATR', ta.average_true_range, ['High', 'Low', 'Close']),
        ('BBH', ta.bollinger_hband, ['Close']),
        ('BBHI', ta.bollinger_hband_indicator, ['Close']),
        ('BBL', ta.bollinger_lband, ['Close']),
        ('BBLI', ta.bollinger_lband_indicator, ['Close']),
        ('BBM', ta.bollinger_mavg, ['Close']),
        ('DCH', ta.donchian_channel_hband, ['Close']),
        ('DCHI', ta.donchian_channel_hband_indicator, ['Close']),
        ('DCL', ta.donchian_channel_lband, ['Close']),
        ('DCLI', ta.donchian_channel_lband_indicator, ['Close']),
        ('KCC', ta.keltner_channel_central, ['High', 'Low', 'Close']),
        ('KCH', ta.keltner_channel_hband, ['High', 'Low', 'Close']),
        ('KCHI', ta.keltner_channel_hband_indicator, ['High', 'Low', 'Close']),
        ('KCL', ta.keltner_channel_lband, ['High', 'Low', 'Close']),
        ('KCLI', ta.keltner_channel_lband_indicator, ['High', 'Low', 'Close']),
        # Trend
        ('AXD', ta.adx, ['High', 'Low', 'Close']),
        ('AXDN', ta.adx_neg, ['High', 'Low', 'Close']),
        ('AXDP', ta.adx_pos, ['High', 'Low', 'Close']),
        ('ARD', ta.aroon_down, ['Close']),
        ('ARU', ta.aroon_up, ['Close']),
        ('CCI', ta.cci, ['High', 'Low', 'Close']),
        ('DPO', ta.dpo, ['Close']),
        ('EMA', ta.trend.ema_indicator, ['Close']),
        ('ICHA', ta.ichimoku_a, ['High', 'Low']),
        ('ICHB', ta.ichimoku_b, ['High', 'Low']),
        ('KST', ta.kst, ['Close']),
        ('KSTS', ta.kst_sig, ['Close']),
        ('MACD', ta.macd, ['Close']),
        ('MACDD', ta.macd_diff, ['Close']),
        ('MACDS', ta.macd_signal, ['Close']),
        ('MI', ta.mass_index, ['High', 'Low']),
        ('TRIX', ta.trix, ['Close']),
        ('VIN', ta.vortex_indicator_neg, ['High', 'Low', 'Close']),
        ('VIP', ta.vortex_indicator_pos, ['High', 'Low', 'Close']),
        # Others
        ('CR', ta.cumulative_return, ['Close']),
        ('DLR', ta.daily_log_return, ['Close']),
        ('DR', ta.daily_return, ['Close']),

    ]

    def __init__(self,
                 indicators: Union[List[str], str, None] = None,
                 lows: Union[List[float], List[int]] = None,
                 highs: Union[List[float], List[int]] = None
                 ):
        """S
        :param columns (optional): A list of indicators to apply as part of the transformation.  Leaving this value
        blank will apply all that are supported by the library.
        """

        self._indicator_names = TAIndicator.indicators if not indicators else indicators
        self._indicators = list(
            map(lambda indicator_name: self._str_to_indicator(indicator_name), indicators))
        self._lows = lows or np.zeros(len(indicators))
        self._highs = highs or np.ones(len(indicators))

    def _str_to_indicator(self, indicator_name: str):
        # Only one result expected
        return next(i for i in TAIndicator.indicators if i[0] == indicator_name)

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        output_space = copy(input_space)
        shape_x, *shape_y = input_space.shape

        output_space.shape = (shape_x + len(self._indicators), *shape_y)

        for i in range(len(self._indicators)):
            output_space.low = np.append(output_space.low, self._lows[i])
            output_space.high = np.append(output_space.high, self._highs[i])

        return output_space

    def transform(self, df: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """
        Will add TAIndicator.indicator columns to DataFrame. Frame must have columns that match indicator parameters,
        e.g. ['High', 'Low', 'Close'], &c...

        :param df: Dataframe with columns matching TA indicators function call
        :param input_space: None is acceptable, doesn't nothing but required by abstract class
        :return:
        """
        for i in range(len(self._indicators)):
            indicator = self._indicators[i]
            name, func, args = indicator
            split = [df[a] for a in args]
            df[name] = func(*split)
        return df


