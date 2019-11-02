from .slippage_model import SlippageModel
from .random_slippage_model import RandomUniformSlippageModel


_registry = {
    'uniform': RandomUniformSlippageModel
}


def get(identifier: str) -> SlippageModel:
    """Gets the `SlippageModel` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `SlippageModel`

    Raises:
        KeyError: if identifier is not associated with any `SlippageModel`
    """
    if identifier not in _registry.keys():
        raise KeyError('Identifier {} is not associated with any `SlippageModel`.'.format(identifier))
    return _registry[identifier]()
