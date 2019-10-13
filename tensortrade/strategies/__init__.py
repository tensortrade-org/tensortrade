from .trading_strategy import TradingStrategy
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
        raise KeyError(f'Identifier {identifier} is not associated with any `TradingStrategy`.')
    return _registry[identifier]
