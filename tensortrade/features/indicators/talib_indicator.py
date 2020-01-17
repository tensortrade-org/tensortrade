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
import pandas as pd

from typing import Union, List

from tensortrade.features import FeatureTransformer


class TAlibIndicator(FeatureTransformer):
    """Adds one or more TAlib indicators to a data frame, based on existing open, high, low, and close column values."""

    def __init__(self, indicators: List[str], lows: Union[List[float], List[int]] = None, highs: Union[List[float], List[int]] = None, **kwargs):
        self._indicator_names = [indicator[0].upper() for indicator in indicators]
        self._indicators = [getattr(talib, name.split('-')[0]) for name in self._indicator_names]
        self._indicator_args = {indicator[0]:indicator[1]['args'] for indicator in indicators}
        self._indicator_params = {indicator[0]: indicator[1]['params'] for indicator in indicators}

        #columns 
        self._ohlcv_cols = ['date','open', 'high', 'low', 'close', 'volume', 'volumeto']
        self._other_cols = [indicator_name.upper() for indicator_name in self._indicator_names if indicator_name != "BBANDS"]
        self._bband_cols = ["bb_upper", "bb_middle", "bb_lower"]

    def init_history(self):
        self._history = pd.DataFrame(columns = self._ohlcv_cols + self._other_cols + self._bband_cols)

    def observation_columns(self):
        return self._other_cols+self._bband_cols

    def transform_spaces(self, low, high):
        new_low, new_high = low.copy(), high.copy()
        return new_low, new_high

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        #concat new observations to the history 
        window_size = len(X)
        self._history = pd.concat([self._history, X], ignore_index=False)

        for idx, indicator in enumerate(self._indicators):
            indicator_name = self._indicator_names[idx]
            indicator_args = [self._history[arg].values for arg in self._indicator_args[indicator_name]]
            indicator_params = self._indicator_params[indicator_name]

            if indicator_name == 'BBANDS':
                upper, middle, lower = indicator(*indicator_args,**indicator_params)

                self._history["bb_upper"] = upper
                self._history["bb_middle"] = middle
                self._history["bb_lower"] = lower
            else:
                try:
                    value = indicator(*indicator_args,**indicator_params)

                    if type(value) == tuple:
                        self._history[indicator_name] = value[0][0]
                    else:
                        self._history[indicator_name] = value

                except:
                    self._history[indicator_name] = indicator(*indicator_args,**indicator_params)[0]

        #return window size of observations 
        return self._history.tail(window_size)
    
    def reset(self):
        self.init_history()
        super().reset()

