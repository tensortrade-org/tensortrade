from .trading_environment import TradingEnvironment


_registry = {}


def get(identifier: str) -> TradingEnvironment:
    """Gets the `TradingEnvironment` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingEnvironment`

    Raises:
        KeyError: if identifier is not associated with any `TradingEnvironment`
    """
    if identifier not in _registry.keys():
        raise KeyError(f'Identifier {identifier} is not associated with any `TradingEnvironment`.')
    return _registry[identifier]
