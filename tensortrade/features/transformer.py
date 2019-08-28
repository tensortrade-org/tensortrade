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
    def transform(self, X: TransformableList, y: TransformableList = None):
        """Transform the data set with the pre-fit model.

        Arguments:
            X: The set of data to transform.
            y (optional): The target output to train on.
        """
        raise NotImplementedError
