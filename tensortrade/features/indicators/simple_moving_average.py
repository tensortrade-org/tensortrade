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


class SimpleMovingAverage(FeatureTransformer):
    """A transformer to add the simple moving average of a column to a feature pipeline."""

    def __init__(self, columns: Union[List[str], str, None] = None, window_size: int = 20, inplace: bool = True, **kwargs):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            window_size (optional): The length of the moving average window. Defaults to 20.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)

        self._window_size = window_size

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)

        for column in self.columns:
            moving_average = X[column].rolling(self._window_size).mean()

            if not self._inplace:
                column = '{}_sma_{}'.format(column, self._window_size)

            args = {}
            args[column] = moving_average

            X = X.assign(**args)

        return X
