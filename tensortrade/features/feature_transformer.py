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

from gym import Space
from typing import List, Union
from abc import ABCMeta, abstractmethod


class FeatureTransformer(object, metaclass=ABCMeta):
    """An abstract feature transformer for use within feature pipelines."""

    def __init__(self, *args, **kwargs):
        pass

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
    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """Get the transformed output space for a given input space.

        Args:
            input_space: A `gym.Space` matching the shape of the pipeline's input.
            column_names: A list of all column names in the input data frame.

        Returns:
            A `gym.Space` matching the shape of the pipeline's output.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """Transform the data set and return a new data frame.

        Arguments:
            X: The set of data to transform.
            input_space: A `gym.Space` matching the shape of the pipeline's input.

        Returns:
            A transformed data frame.
        """
        raise NotImplementedError
