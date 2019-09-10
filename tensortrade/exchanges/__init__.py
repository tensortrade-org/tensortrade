from .instrument_exchange import InstrumentExchange

from . import live
from . import simulated

_registry = {
    'ccxt': live.CCXTExchange,
    'simulated': simulated.SimulatedExchange,
    'fbm': simulated.FBMExchange,
    'gan': simulated.GANExchange
}


def get(identifier):
    if identifier not in _registry.keys():
        raise KeyError('Identifier is not a registered identifier.')
    return _registry[identifier]