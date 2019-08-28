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
# limitations under the License

import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.externals.joblib import Parallel, delayed

from tensortrade.features.transformer import TransformableList


class PDFeatureUnion(FeatureUnion):
    """A utility class for unioning pipelines while maintaining underlying pandas.DataFrame data structure."""

    def transform(self, X: TransformableList):
        """Transform the data set with the fit model.

        Arguments:
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

        Arguments:
            X: The set of data to train the model on and transform.
            y (optional): The target output to train with.

        Returns:
            A transformed set of features.
        """
        return self.fit(X, y).transform(X)
