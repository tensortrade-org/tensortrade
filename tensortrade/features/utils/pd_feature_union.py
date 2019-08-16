import pandas as pd

from typing import Union, List
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals.joblib import Parallel, delayed

from tensortrade.features.transformer import TransformableList


class PDFeatureUnion(FeatureUnion):
    """A utility class for unioning pipelines while maintaining underlying Pandas.DataFrame data structure."""

    def transform(self, X: TransformableList):
        """Transform the data set with the fit model.

        Args:
            X: The set of data to transform.

        Returns:
            A transformed set of features.
        """

        def inner_transform_with_weights(transformer, X, y, weight):
            transformed = transformer.transform(X)
            return transformed if weight is None else transformed * weight

        transformed_X = Parallel(n_jobs=self.n_jobs)(delayed(inner_transform_with_weights)(trans, weight, X)
                                                     for _, trans, weight in self._iter())

        return pd.concat(transformed_X, axis=1, join='inner')

    def fit_transform(self, X: TransformableList, y: TransformableList = None):
        """Fit the model to the data set, then transform the data set with the fit model.

        Args:
            X: The set of data to train the model on and transform.
            y (optional): The target output to train with.

        Returns:
            A transformed set of features.
        """
        return self.fit(X, y).transform(X)
