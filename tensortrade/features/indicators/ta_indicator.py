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

    def __init__(self,
                 indicators: Union[List[str], str, None] = None,
                 lows: Union[List[float], List[int]] = None,
                 highs: Union[List[float], List[int]] = None
                 ):

        if isinstance(indicators, str):
            indicators = [indicators]

        self._indicator_names = indicators
        self._indicators = list(
            map(lambda indicator_name: self._str_to_indicator(indicator_name), indicators))
        self._lows = lows or np.zeros(len(indicators))
        self._highs = highs or np.ones(len(indicators))

    def _str_to_indicator(self, indicator_name: str):
        return getattr(ta, indicator_name.lower())

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

        :param df: Dataframe with columns matching TA indicators function call. Not case sensitive
        :param input_space: None is acceptable, does nothing but required by abstract class
        :return:
        """
        for i in range(len(self._indicators)):
            indicator = self._indicators[i]
            indicator_name = self._indicator_names[i]
            params = {}
            for param in indicator.__code__.co_varnames:
                if param in ["df", "open", "high", "low", "close", "volume"]:
                    if param == "df":
                        params[param] = df
                    else:
                        for column in df.columns:
                            if column.lower() == param:
                                params[param] = df[column]
            df[indicator_name] = indicator(**params)
        return df


