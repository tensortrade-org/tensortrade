from .trading_environment import TradingEnvironment

from . import render

_registry = {
    'basic': {
        'exchange': 'simulated',
        'action_scheme': 'discrete',
        'reward_scheme': 'simple'
    }
}


def get(identifier: str) -> TradingEnvironment:
    """Gets the `TradingEnvironment` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingEnvironment`

    Raises:
        KeyError: if identifier is not associated with any `TradingEnvironment`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `TradingEnvironment`.'.format(identifier))
    return TradingEnvironment(**_registry[identifier])
