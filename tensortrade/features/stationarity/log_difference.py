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


class LogDifference(FeatureTransformer):
    """A transformer for taking the logarithmic difference of values within a feature pipeline."""

    def __init__(self,
                 columns: Union[List[str], str, None] = None,
                 inplace: bool = True):
        """
        Arguments:
            columns (optional): A list of column names to difference.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)

        for column in self.columns:
            diffed_series = np.log(X[column]) - np.log(X[column].shift(1))
            diffed_series = diffed_series.fillna(method='bfill')

            if not self._inplace:
                column = '{}_log_diff'.format(column)

            args = {}
            args[column] = diffed_series

            X = X.assign(**args)

        return X.iloc[-len(X):]
