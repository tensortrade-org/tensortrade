from .trading_strategy import TradingStrategy
from .stable_baselines_strategy import StableBaselinesTradingStrategy
from .tensorforce_trading_strategy import TensorforceTradingStrategy


_registry = {}


def get(identifier: str) -> TradingStrategy:
    """Gets the `TradingStrategy` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingStrategy`

    Raises:
        KeyError: if identifier is not associated with any `TradingStrategy`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `TradingStrategy`.'.format(identifier))
    return _registry[identifier]
