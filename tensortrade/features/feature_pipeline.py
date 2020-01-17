
import pandas as pd
import numpy as np

from typing import List, Union

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

    @property
    def observation_columns(self) -> pd.DataFrame:
        ret = [] 
        for transformer in self._steps:
            ret+= transformer.observation_columns()
        return ret 
        
    def reset(self):
        """Reset all transformers within the feature pipeline."""
        for transformer in self._steps:
            transformer.reset()

    def transform_spaces(self, low, high):
        """Transforms bounding space same as observations to reflect the post processed data""" 
        for transformer in self._steps: 
            low, high = transformer.transform_spaces(low, high)
        return low,high 

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
