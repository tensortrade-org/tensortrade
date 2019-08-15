import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn import Pipeline
from sklearn.utils import check_array


class FeaturePipeline(object, metaclass=ABCMeta):
    """An abstract base class for feature transformation pipelines."""

    def __init__(self, pipeline: Pipeline, dtype: type = np.float16):
        """
        Args:
            pipeline: An `sklearn.Pipeline` instance of feature transformations.
        """
        self._pipeline = pipeline
        self._dtype = dtype

    @property
    def pipeline(self):
        """An `sklearn.Pipeline` instance of feature transformations."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit_transform(self, observation: pd.DataFrame) -> np.ndarray:
        """Fit and apply the pipeline of feature transformations to an observation frame.

        Args:
            observation: A `pandas.DataFrame` corresponding to an observation within a `TradingEnvironment`.

        Returns:
            A `numpy.ndarray` of features.

        Raises:
            ValueError: In the case that an invalid observation frame has been input.
        """
        try:
            features = check_array(observation, dtype=self._dtype)
        except ValueError as e:
            raise ValueError(f'Invalid observation frame passed to feature pipeline: {e}')

        features = self._pipeline.fit_transform(features)

        if isinstance(features, pd.DataFrame):
            return features.values

        return features
