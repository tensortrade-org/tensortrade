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

from typing import Union, List, Tuple
from sklearn.preprocessing import StandardScaler

from tensortrade.features.transformer import Transformer, TransformableList


class StandardNormalizer(Transformer):
    """A transformer for normalizing values within a feature pipeline by removing the mean and scaling to unit variance."""

    def __init__(self, columns: Union[List[str], str] = None):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
        """
        self._columns = columns
        self._scaler = StandardScaler()

    def transform(self, X: TransformableList, y: TransformableList = None):
        if self._columns is None:
            return self._scaler.fit_transform(X, y)

        return self._scaler.fit_transform(X[self._columns], y)
