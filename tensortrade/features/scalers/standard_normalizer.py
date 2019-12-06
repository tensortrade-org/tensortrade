

from math import log
import pandas as pd
import numpy as np

from gym import Space
from copy import copy
from typing import Union, List, Tuple
from loguru import logger
from tensortrade.features.feature_transformer import FeatureTransformer


class StandardNormalizer(FeatureTransformer):
    """A transformer for normalizing values within a feature pipeline by removing the mean and scaling to unit variance."""

    def __init__(self, columns: Union[List[str], str, None] = None, feature_min=0, feature_max=1, inplace=True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            feature_min (optional): The minimum value in the range to scale to.
            feature_max (optional): The maximum value in the range to scale to.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)

        self._feature_min = feature_min
        self._feature_max = feature_max

        if feature_min >= feature_max:
            raise ValueError("feature_min must be less than feature_max")

        self._history = {}

    def reset(self):
        self._history = {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)
        
        for column in self.columns:
            if self._inplace == True:
                X[column] = (X[column] - X[column].mean())/X[column].std()
            else:
                X[f"{column}_scaled"] = (X[column] - X[column].mean())/X[column].std()
            
        return X.dropna()