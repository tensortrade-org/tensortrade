from ccxt import coinbasepro
from .instrument_exchange import InstrumentExchange

from . import live
from . import simulated

_registry = {
    'ccxt': live.CCXTExchange(exchange=coinbasepro()),
    'simulated': simulated.SimulatedExchange(),
    'fbm': simulated.FBMExchange(),
    'gan': simulated.GANExchange()
}


def get(identifier: str) -> InstrumentExchange:
    """Gets the `InstrumentExchange` that matches with the identifier.

    As a caution, when exchanges that require a data frame are instantiated by
    this function, the data frame is set as None and must be set at a later
    point in time for the exchange to work.

    Arguments:
        identifier: The identifier for the `InstrumentExchange`

    Raises:
        KeyError: if identifier is not associated with any `InstrumentExchange`
    """
    if identifier not in _registry.keys():
        raise KeyError(f'Identifier {identifier} is not associated with any `InstrumentExchange`.')
    return _registry[identifier]
