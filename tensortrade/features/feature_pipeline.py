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

from gym import Space
from typing import List, Union, Callable

from tensortrade import Component
from .feature_transformer import FeatureTransformer


class FeaturePipeline(Component):
    """An pipeline for transforming observation data frames into features for learning."""
    registered_name = "features"

    def __init__(self, steps: List[FeatureTransformer], **kwargs):
        """
        Arguments:
            steps: A list of feature transformations to apply to observations.
        """
        self._steps = steps
        self._dtype: Union[type, str] = self.default('dtype', np.float32, kwargs)

    @property
    def steps(self) -> List[FeatureTransformer]:
        """A list of feature transformations to apply to observations."""
        return self._steps

    @steps.setter
    def steps(self, steps: List[FeatureTransformer]):
        self._steps = steps

    def reset(self):
        """Reset all transformers within the feature pipeline."""
        for transformer in self._steps:
            transformer.reset()

    def _transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Utility method for transforming observations via a list of `FeatureTransformer` objects."""
        for transformer in self._steps:
            observations = transformer.transform(observations)

        return observations

    def transform(self, observation: pd.DataFrame) -> pd.DataFrame:
        """Apply the pipeline of feature transformations to an observation frame.

        Arguments:
            observation: A `pandas.DataFrame` corresponding to an observation within a `TradingEnvironment`.

        Returns:
            A `pandas.DataFrame` of features corresponding to an input oversvation.

        Raises:
            ValueError: In the case that an invalid observation frame has been input.
        """
        obs = observation.copy(deep=True)
        features = self._transform(obs)

        if not isinstance(features, pd.DataFrame):
            raise ValueError("A FeaturePipeline must transform a pandas.DataFrame into another pandas.DataFrame.\n \
                               Expected return type: {} `\n \
                               Actual return type: {}.".format(type(pd.DataFrame([])), type(features)))

        return features
