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
from typing import Union, List, Tuple

from tensortrade.features.feature_transformer import FeatureTransformer


class StandardNormalizer(FeatureTransformer):
    """A transformer for normalizing values within a feature pipeline by removing the mean and scaling to unit variance."""

    def __init__(self, columns: Union[List[str], str, None] = None, inplace=True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)

        self._history = {}

    def reset(self):
        self._history = {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)

        for column in self.columns:
            if not self._inplace:
                column = '{}_standardize'.format(column)

            args = {}
            args[column] = (X[column] - X[column].mean()) / X[column].std()

            X = X.assign(**args)

        return X
