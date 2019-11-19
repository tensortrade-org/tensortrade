import ccxt
import pandas as pd

from datetime import datetime

from .exchange import Exchange

from . import live
from . import simulated


_registry = {
    'simulated': simulated.SimulatedExchange,
    'fbm': simulated.FBMExchange,
    'gan': simulated.GANExchange
}


def get(identifier: str) -> Exchange:
    """Gets the `Exchange` that matches with the identifier.

    As a caution, when exchanges that require a data frame are instantiated by
    this function, the data frame is set as None and must be set at a later
    point in time for the exchange to work.

    Arguments:
        identifier: The identifier for the `Exchange`

    Raises:
        KeyError: if identifier is not associated with any `Exchange`
    """
    if identifier in _registry.keys():
        if identifier == 'simulated':
            data_url = "http://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_1h.csv"
            data = pd.read_csv(data_url, skiprows=[0])[::-1]
            data = data.get(['Open', 'High', 'Low', 'Close', 'Volume BTC'])
            data = data.rename({'Volume BTC': 'volume'}, axis=1)
            data = data.rename({name: name.lower() for name in data.columns}, axis=1)
            return _registry['simulated'](data_frame=data)

        return _registry[identifier]()

    if identifier in ccxt.exchanges:
        return live.CCXTExchange(exchange=identifier)

    raise KeyError('Identifier {} is not associated with any `Exchange`.'.format(identifier))
