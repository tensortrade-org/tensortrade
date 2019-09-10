from .trading_strategy import TradingStrategy
from .tensorforce_trading_strategy import TensorforceTradingStrategy


_registry = {}


def get(identifier):
    if identifier not in _registry.keys():
        raise KeyError('Identifier is not a registered identifier.')
    return _registry[identifier]
