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

from gym import Space
from typing import Union, List, Tuple

from tensortrade.features.feature_transformer import FeatureTransformer


class SimpleMovingAverage(FeatureTransformer):
    """A transformer to get the simple moving average of one or more columns in a data frame."""

    def __init__(self, columns: Union[List[str], str, None] = None):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
        """
        self.columns = columns

    def transform_space(self, input_space: Space) -> Space:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
