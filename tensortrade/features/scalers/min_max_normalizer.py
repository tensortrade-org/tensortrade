import pandas as pd
import numpy as np

from typing import Union, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn import Pipeline

from tensortrade.features.transformer import Transformer, TransformableList


class MinMaxNormalizer(Transformer):
    """A transformer for normalizing values within a feature pipeline by the column-wise extrema."""

    def __init__(self, feature_range: Tuple[int, int] = (0, 1), columns: Union[List[str], str] = None):
        """
        Args:
            feature_range (optional): A tuple containing the new `(minimum, maximum)` values to scale to.
            columns (optional): A list of column names to normalize.
        """
        self._columns = columns
        self._scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, X: TransformableList, y: TransformableList = None):
        if self._columns is None:
            return self._scaler.fit(X, y)

        return self._scaler.fit(X[self._columns], y)

    def transform(self, X: TransformableList):
        if self._columns is None:
            return self._scaler.transform(X)

        return self._scaler.transform(X[self._columns])
