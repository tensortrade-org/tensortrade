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
        self._indicator_names = [indicator[0].upper() for indicator in indicators]
        self._indicator_args = {indicator[0]:indicator[1]['args'] for indicator in indicators}
        self._indicator_params = {indicator[0]: indicator[1]['params'] for indicator in indicators}
        self._indicators = [getattr(talib, name.split('-')[0]) for name in self._indicator_names]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for idx, indicator in enumerate(self._indicators):
            indicator_name = self._indicator_names[idx]
            indicator_args = [X[arg].values for arg in self._indicator_args[indicator_name]]
            indicator_params = self._indicator_params[indicator_name]

            if indicator_name == 'BBANDS':
                upper, middle, lower = indicator(*indicator_args,**indicator_params)

                X["bb_upper"] = upper
                X["bb_middle"] = middle
                X["bb_lower"] = lower
            else:
                try:
                    value = indicator(*indicator_args,**indicator_params)

                    if type(value) == tuple:
                        X[indicator_name] = value[0][0]
                    else:
                        X[indicator_name] = value

                except:
                    X[indicator_name] = indicator(*indicator_args,**indicator_params)[0]

        return X
