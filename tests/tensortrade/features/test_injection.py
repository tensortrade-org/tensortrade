import pandas as pd

from gym import Space
from typing import List
from itertools import repeat


from tensortrade import TradingContext
from tensortrade.features.feature_transformer import FeatureTransformer
from tensortrade.features.feature_pipeline import FeaturePipeline


class Identity(FeatureTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


def test_injects_feature_transformation_with_context():

    config = {
        'features': {
            'shape': (90, 70)
        }
    }

    with TradingContext(**config):

        transformer = Identity()

        assert hasattr(transformer.context, 'shape')
        assert transformer.context.shape == (90, 70)


def test_injects_feature_pipeline_with_context():

    config = {
        'features': {
            'shape': (90, 70)
        }
    }

    with TradingContext(**config):

        steps = list(repeat(Identity(), 5))
        pipeline = FeaturePipeline(steps)
        assert hasattr(pipeline.context, 'shape')
        assert pipeline.context.shape == (90, 70)
