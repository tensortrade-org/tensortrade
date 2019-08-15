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
        """
        Args:
            X: The set of data to train the transformer on.
            y (optional): The target output to train with.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: TransformableList, y: TransformableList = None):
        """
        Args:
            X: The set of data to transform.
            y (optional): The target output to evaluate with.
        """
        raise NotImplementedError
