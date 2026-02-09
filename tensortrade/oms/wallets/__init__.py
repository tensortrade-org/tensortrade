from .portfolio import Portfolio
from .wallet import Wallet

_registry = {}


def get(identifier: str) -> Portfolio:
    """Gets the `TradingStrategy` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingStrategy`

    Raises:
        KeyError: if identifier is not associated with any `TradingStrategy`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            f"Identifier {identifier} is not associated with any `TradingStrategy`."
        )
    return _registry[identifier]
