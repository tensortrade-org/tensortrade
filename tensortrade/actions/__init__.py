from .action_scheme import ActionScheme, DTypeString, TradeActionUnion
from .continuous_actions import ContinuousActions
from .discrete_actions import DiscreteActions
from .multi_discrete_actions import MultiDiscreteActions


_registry = {
    'continuous': ContinuousActions,
    'discrete': DiscreteActions,
    'multi-discrete': MultiDiscreteActions,
}


def get(identifier: str) -> ActionScheme:
    """Gets the `ActionScheme` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `ActionScheme`

    Raises:
        KeyError: if identifier is not associated with any `ActionScheme`
    """
    if identifier not in _registry.keys():
        raise KeyError(f'Identifier {identifier} is not associated with any `ActionScheme`.')

    return _registry[identifier]()
