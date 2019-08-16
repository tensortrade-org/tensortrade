import numpy as np
import pandas as pd

from typing import Union
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

TransformableList = Union[np.ndarray, pd.DataFrame]


class Transformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """An abstract base class for transformers within feature pipelines."""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, X: TransformableList, y: TransformableList = None):
        """Fit the model to the data set, if necessary, else return self.

        Args:
            X: The set of data to train the model on.
            y (optional): The target output to train with.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: TransformableList):
        """Transform the data set with the fit model.

        Args:
            X: The set of data to transform.
        """
        raise NotImplementedError
