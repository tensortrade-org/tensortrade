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


class ComparisonNormalizer(FeatureTransformer):
    """
    A transformer for normalizing values within a feature pipeline using a comparison column.
    This is useful for normalizing values against a base column, for example technical analysis indicators
    such as moving averages or bollinger bands against the close price. Can also be used with open, high, low, close.
    This normalizer converts by default to a scale of [0,1]. A value equal to the comparison column will receive a value of 0.5.
    Value above the comparison column will be in the range 0.5-1 and values below will be in the range 0-0.5.
    Items which are more than 100% of the comparison column will be clipped to 1 and items which are of opposite sign (negative) will be clipped to 0.
    """

    def __init__(self,
                 columns: Union[List[str], str, None] = None,
                 comparison_column: str = 'close',
                 feature_min: float = 0,
                 feature_max: float = 1,
                 inplace: bool = True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            comparison_column (optional): The column name of the price column on which normalization will be performed. Defaults to 'close'.
            feature_min (optional): The minimum `float` in the range to scale to. Defaults to 0.
            feature_max (optional): The maximum `float` in the range to scale to. Defaults to 1.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)

        self._comparison_column = comparison_column
        self._feature_min = feature_min
        self._feature_max = feature_max

        if feature_min >= feature_max:
            raise ValueError("feature_min must be less than feature_max")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)

        if self._comparison_column not in X:
            raise ValueError("Unable to find column {}".format(self._comparison_column))

        for column in self.columns:
            normalized_column = X[column] / (2 * X[self._comparison_column])

            if not self._inplace:
                column = '{}_price_{}_{}'.format(column, self._feature_min, self._feature_max)

            args = {}
            args[column] = normalized_column

            X = X.assign(**args)

        return X
