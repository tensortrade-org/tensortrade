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
#
# Reference Source: Marcos Lopez De Prado - Advances in Financial Machine Learning
#                   Chapter 5 (Pg. 82) - Fractionally Differentiated Features

import pandas as pd
import numpy as np

from gym import Space
from copy import copy
from typing import Union, List, Tuple

from tensortrade.features.feature_transformer import FeatureTransformer


class FractionalDifference(FeatureTransformer):
    """A transformer for differencing values within a feature pipeline by a fractional order."""

    def __init__(self,
                 all_column_names: List[str] = None,
                 columns: Union[List[str], str, None] = None,
                 difference_order: float = 0.5,
                 difference_threshold: float = 1e-1,
                 inplace: bool = True):
        """
        Arguments:
            all_column_names: A list of all column names in the data frame.
            columns (optional): A list of column names to difference.
            difference_order (optional): The fractional difference order. Defaults to 0.5.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        self._all_column_names = all_column_names
        self._difference_order = difference_order
        self._difference_threshold = difference_threshold
        self._inplace = inplace
        self.columns = columns

        self._history = None

        if self._all_column_names is None:
            raise ValueError('FractionalDifference requires passing a list of `all_column_names` from the observation data frame.\n \
                              This is necessary to correctly transform the `observation_space`.')

        if not isinstance(self._all_column_names, list):
            self._all_column_names = list(self._all_column_names)

    def reset(self):
        self._history = None

    def transform_space(self, input_space: Space) -> Space:
        if self._inplace:
            return input_space

        output_space = copy(input_space)
        columns = self.columns or self._all_column_names

        shape_x, *shape_y = input_space.shape
        output_space.shape = (shape_x + len(columns), *shape_y)

        for column in columns:
            column_index = self._all_column_names.index(column)
            low, high = input_space.low[column_index], input_space.high[column_index]

            output_space.low = np.append(output_space.low - output_space.high, low)
            output_space.high = np.append(output_space.high, high)

        return output_space

    def _difference_weights(self, size: int):
        weights = [1.0]

        for k in range(1, size):
            weight = -weights[-1] / k * (self._difference_order - k + 1)
            weights.append(weight)

        return np.array(weights[::-1]).reshape(-1, 1)

    def _fractional_difference(self, series: pd.Series):
        """Computes fractionally differenced series, with an increasing window width.

        Args:
            series: A `pandas.Series` to difference by self._difference_order with self._difference_threshold.

        Returns:
            The fractionally differenced series.
        """
        weights = self._difference_weights(len(series))

        weight_sums = np.cumsum(abs(weights))
        weight_sums /= weight_sums[-1]

        skip_weights = len(weight_sums[weight_sums > self._difference_threshold])

        curr_series = series.dropna()
        diff_series = pd.Series(index=series.index)

        for current_index in range(skip_weights, curr_series.shape[0]):
            index = curr_series.index[current_index]

            if not np.isfinite(curr_series.loc[index]):
                continue

            diff_series[index] = np.dot(
                weights[-(current_index + 1):, :].T, curr_series.loc[:index])[0]

        return diff_series

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._history is None:
            self._history = X.copy()
        else:
            self._history = self._history.append(X, ignore_index=True)

        if len(self._history) > len(X):
            self._history = self._history.iloc[-len(X) + 1:]

        if self.columns is None:
            self.columns = list(X.columns)

        for column in self.columns:
            diffed_series = self._fractional_difference(self._history[column])

            if self._inplace:
                X[column] = diffed_series.fillna(method='bfill')
            else:
                column_name = '{}_diff_{}'.format(column, self._difference_order)
                X[column_name] = diffed_series.fillna(method='bfill')

        return X.iloc[-len(X):]
