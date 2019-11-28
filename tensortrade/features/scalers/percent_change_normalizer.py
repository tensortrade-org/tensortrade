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

import pandas as pd
import numpy as np

from gym import Space
from copy import copy
from typing import Union, List, Tuple, Dict

from tensortrade.features.feature_transformer import FeatureTransformer


class PercentChangeNormalizer(FeatureTransformer):
    """
    A transformer for normalizing values within a feature pipeline by the percent change of the previous value.
    This would be used in cases where you want to generalize away from values where specific values have a meaning.
    For example, if you are thinking about this on the close price, it has an advantage and a disadvantage:
        - advantage: the price jump from 100 to 101 will be normalized the same as the price jump from 101 to 102.1
        - disadvantage: the system will have a harder time with action dependent on specific values, like the price
                        is constantly attracted to 100.
    """

    def __init__(self,
                 columns: Union[List[str], str, None] = None,
                 feature_min: float = 0,
                 feature_max: float = 1,
                 inplace: bool = True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            feature_min (optional): The minimum `float` in the range to scale to. Defaults to 0.
            feature_max (optional): The maximum `float` in the range to scale to. Defaults to 1.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)

        self._feature_min = feature_min
        self._feature_max = feature_max

        if feature_min >= feature_max:
            raise ValueError("feature_min must be less than feature_max")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)

        for column in self.columns:
            feature_midpoint = (self._feature_max + self._feature_min) / 2
            feature_scale = 1

            if self._feature_max - self._feature_min < 1:
                feature_scale = 1 / (self._feature_max - self._feature_min)

            normalized_column = feature_scale * X[column].pct_change() + feature_midpoint
            normalized_column = normalized_column.fillna(feature_midpoint)
            normalized_column = normalized_column.clip(lower=self._feature_min,
                                                       upper=self._feature_max)

            if not self._inplace:
                column = '{}_price_{}_{}'.format(column, self._feature_min, self._feature_max)

            args = {}
            args[column] = normalized_column

            X = X.assign(**args)

        return X
