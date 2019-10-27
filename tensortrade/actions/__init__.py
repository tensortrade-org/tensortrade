from .action_strategy import ActionStrategy, DTypeString, TradeActionUnion
from .continuous_action_strategy import ContinuousActionStrategy
from .discrete_action_strategy import DiscreteActionStrategy
from .multi_discrete_action_strategy import MultiDiscreteActionStrategy
from .target_stop_action_strategy import TargetStopActionStrategy


_registry = {
    'continuous': ContinuousActionStrategy(),
    'discrete': DiscreteActionStrategy(),
    'multi-discrete': MultiDiscreteActionStrategy(instrument_symbols=['BTC', 'ETH']),
    'target-stop': TargetStopActionStrategy,
}


def get(identifier: str) -> ActionStrategy:
    """Gets the `ActionStrategy` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `ActionStrategy`

    Raises:
        KeyError: if identifier is not associated with any `ActionStrategy`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `ActionStrategy`.'.format(identifier))
    return _registry[identifier]
