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


class MinMaxNormalizer(FeatureTransformer):
    """A transformer for normalizing values within a feature pipeline by the column-wise extrema."""

    def __init__(self,
                 columns: Union[List[str], str, None] = None,
                 feature_min=0,
                 feature_max=1,
                 inplace=True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            feature_min (optional): The minimum value in the range to scale to.
            feature_max (optional): The maximum value in the range to scale to.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        self._feature_min = feature_min
        self._feature_max = feature_max
        self._inplace = inplace
        self.columns = columns

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        if self._inplace:
            return input_space

        output_space = copy(input_space)

        shape_x, *shape_y = input_space.shape

        columns = self.columns or range(len(shape_x))

        output_space.shape = (shape_x + len(columns), *shape_y)

        for _ in columns:
            output_space.low = np.append(output_space.low, self._feature_min)
            output_space.high = np.append(output_space.high, self._feature_max)

        return output_space

    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.columns)

        for idx, column in enumerate(self.columns):
            low = input_space.low[idx]
            high = input_space.high[idx]

            scale = (self._feature_max - self._feature_min) + self._feature_min
            normalized_column = (X[column] - low) / (high - low + 1E-9) * scale

            if self._inplace:
                X[column] = normalized_column
            else:
                column_name = '{}_minmax_{}_{}'.format(column, self._feature_min, self._feature_max)
                X[column_name] = normalized_column

        return X
