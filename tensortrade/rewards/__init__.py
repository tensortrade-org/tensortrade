from .reward_strategy import RewardStrategy
from .simple_profit_strategy import SimpleProfitStrategy


_registry = {
    'simple': SimpleProfitStrategy
}


def get(identifier):
    if identifier not in _registry.keys():
        raise KeyError('Identifier is not a registered identifier.')
    return _registry[identifier]
