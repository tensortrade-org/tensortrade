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

import pandas as pd
import numpy as np

from copy import copy
from typing import List, Union
from abc import ABCMeta, abstractmethod

from tensortrade import Component


class FeatureTransformer(Component, metaclass=ABCMeta):
    """An abstract feature transformer for use within feature pipelines."""

    registered_name = "features"

    def __init__(self, columns: Union[List[str], str, None] = None, inplace: bool = True, **kwargs):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        self.columns = self.default('columns', columns)
        self._inplace = self.default('inplace', inplace)

    @property
    def columns(self) -> List[str]:
        return self._columns

    @columns.setter
    def columns(self, columns=Union[List[str], str]):
        self._columns = columns

        if isinstance(self._columns, str):
            self._columns = [self._columns]

    def reset(self):
        """Optionally implementable method for resetting stateful transformers."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data set and return a new data frame.

        Arguments:
            X: The set of data to transform.

        Returns:
            A transformed data frame.
        """
        raise NotImplementedError
