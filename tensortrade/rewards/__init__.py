from .reward_strategy import RewardStrategy
from .simple_profit_strategy import SimpleProfitStrategy


_registry = {
    'simple': SimpleProfitStrategy()
}


def get(identifier: str) -> RewardStrategy:
    """Gets the `RewardStrategy` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `RewardStrategy`

    Raises:
        KeyError: if identifier is not associated with any `RewardStrategy`
    """
    if identifier not in _registry.keys():
        raise KeyError(f'Identifier {identifier} is not associated with any `RewardStrategy`.')
    return _registry[identifier]
