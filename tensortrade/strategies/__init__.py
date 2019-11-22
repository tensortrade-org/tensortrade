import importlib

from .trading_strategy import TradingStrategy

if importlib.util.find_spec("stable_baselines") is not None:
    from .stable_baselines_strategy import StableBaselinesTradingStrategy
if importlib.util.find_spec("tensorforce") is not None:
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
