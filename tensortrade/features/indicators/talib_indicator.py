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

import talib
import numpy as np
import pandas as pd

from gym import Space
from copy import copy
from abc import abstractmethod
from typing import Union, List, Callable

from tensortrade.features import FeatureTransformer


class TAlibIndicator(FeatureTransformer):
    """Adds one or more TAlib indicators to a data frame, based on existing open, high, low, and close column values."""

    def __init__(self, indicators: List[str], lows: Union[List[float], List[int]] = None, highs: Union[List[float], List[int]] = None, **kwargs):
        self._indicator_names, self._indicator_values = self.parse_indicators(indicators)
        self._indicators = list(
            map(lambda indicator_name: self._str_to_indicator(indicator_name), self._indicator_names))

        self._history = self.init_history()
        self._window_size = kwargs.get("window_size", 10)
        self._lows = lows or np.zeros(len(self.hist_cols))
        self._highs = highs or np.ones(len(self.hist_cols))            

    def init_history(self):
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        other_cols = [indicator_name.upper() for indicator_name in self._indicator_names if indicator_name != "BBANDS"]
        bband_cols = ["bb_upper", "bb_middle", "bb_lower"]
        self.hist_cols = other_cols + bband_cols
        return pd.DataFrame(columns = ohlcv_cols + self.hist_cols)

    def parse_indicators(self, indicators):
        indicators_list, values_list = [], {}
        for element in indicators:
            i,j = element[0], element[1]
            indicators_list.append(i)
            values_list[i] = j
        return indicators_list, values_list

    def _str_to_indicator(self, indicator_name: str):
        return getattr(talib, indicator_name.upper())

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        output_space = copy(input_space)
        *shape_x, shape_y = input_space.shape

        output_space.shape = (*shape_x , shape_y + len(self.hist_cols))
        #add new lows and highs for indicators in obs space
        for i in range(len(self.hist_cols)):
            new_lows = np.array([self._lows[i]] * self._window_size)[:,None]
            new_highs = np.array([self._highs[i]] * self._window_size)[:,None]
            output_space.low = np.concatenate((output_space.low, new_lows), axis=1)
            output_space.high = np.concatenate((output_space.high, new_highs), axis=1)

        return output_space

    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        #append new obs to history 
        self._history = self._history.append(X.loc[X.index[-1], :])
        self._history = self._history.fillna(0)

        #loop over and apend each indicator to the last row of the history
        for i in range(len(self._indicators)):
            indicator_name = self._indicator_names[i]
            indicator = self._indicators[i]
            indicator_args = [self._history[val].values for val in self._indicator_values[indicator_name]]

            if indicator_name.upper() == 'MACD':
                macd, macd_signal, _ = indicator(*indicator_args)
                self._history[indicator_name.upper()] = macd

            elif indicator_name.upper() == 'BBANDS':
                upper, middle, lower = indicator(*indicator_args)
                self._history["bb_upper"] = upper
                self._history["bb_middle"] = middle
                self._history["bb_lower"] = lower

            else:
                self._history[indicator_name.upper()] = indicator(*indicator_args)
                    
        #we must return number of entries equal to the window size
        self._history = self._history.fillna(0) 
        ret = self._history.tail(self._window_size).copy()
        if len(ret) < 10:
            padding = np.zeros((self._window_size-len(self._history), len(self._history.columns)))
            ret = pd.concat([pd.DataFrame(padding, columns=self._history.columns), self._history.iloc[:,:]], ignore_index=False)
        return ret

    def reset(self):
        self._history = self.init_history()
        super().reset()

