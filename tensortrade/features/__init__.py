from .feature_pipeline import FeaturePipeline
from .feature_transformer import FeatureTransformer

from . import indicators
from . import scalers
from . import stationarity


_registry = {}


def get(identifier: str) -> FeaturePipeline:
    """Gets the `FeaturePipeline` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `FeaturePipeline`

    Raises:
        KeyError: if identifier is not associated with any `FeaturePipeline`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `FeaturePipeline`.'.format(identifier))
    return _registry[identifier]
