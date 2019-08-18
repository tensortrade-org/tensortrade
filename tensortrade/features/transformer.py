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

import numpy as np
import pandas as pd

from typing import Union
from abc import abstractmethod
from sklearn.base import TransformerMixin

TransformableList = Union[np.ndarray, pd.DataFrame]


class Transformer(TransformerMixin):
    """An abstract transformer for use within feature pipelines."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, X: TransformableList, y: TransformableList = None):
        """Fit the model to the data set, if necessary, else return self.

        Args:
            X: The set of data to train the model on.
            y (optional): The target output to train on.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: TransformableList):
        """Transform the data set with the pre-fit model.

        Args:
            X: The set of data to transform.
        """
        raise NotImplementedError
