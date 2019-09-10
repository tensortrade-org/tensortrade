from .action_strategy import ActionStrategy, DTypeString, TradeActionUnion
from .continuous_action_strategy import ContinuousActionStrategy
from .discrete_action_strategy import DiscreteActionStrategy


_registry = {
    'continuous': ContinuousActionStrategy,
    'discrete': DiscreteActionStrategy
}


def get(identifier):
    if identifier not in _registry.keys():
        raise KeyError('Identifier is not a registered identifier.')
    return _registry[identifier]
