from .slippage_model import SlippageModel
from .random_slippage_model import RandomUniformSlippageModel


_registry = {
    'uniform': RandomUniformSlippageModel
}


def get(identifier):
    if identifier not in _registry.keys():
        raise KeyError('Identifier is not a registered identifier.')
    return _registry[identifier]
