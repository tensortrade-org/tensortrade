from .reward_strategy import RewardStrategy
from .simple_profit_strategy import SimpleProfitStrategy
from .risk_adjusted_return_strategy import RiskAdjustedReturnStrategy

_registry = {
    'simple': SimpleProfitStrategy,
    'risk-adjusted': RiskAdjustedReturnStrategy
}


def get(identifier: str) -> RewardStrategy:
    """Gets the `RewardStrategy` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `RewardStrategy`

    Raises:
        KeyError: if identifier is not associated with any `RewardStrategy`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `RewardStrategy`.'.format(identifier))
    return _registry[identifier]()
