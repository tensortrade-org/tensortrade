import numpy as np
import pandas as pd

from typing import Union, List

from tensortrade.features.transformer import Transformer, TransformableList


class ColumnSelector(Transformer):
    """A transformer for selecting named columns within a feature pipeline."""

    def __init__(self, columns: Union[List[str], str]):
        """
        Args:
            columns: A list of column keys to be selected from the pipeline.
        """
        self._columns = columns

    def fit(self, X: TransformableList, y: TransformableList = None):
        return self

    def transform(self, X: TransformableList):
        return X[self._columns]
