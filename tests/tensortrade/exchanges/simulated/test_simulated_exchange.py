

from tensortrade import TradingContext
from tensortrade.exchanges.simulated import SimulatedExchange


config = {
        'base_instrument': 'USD',
        'products': 'ETH',
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_base_instrument():

    with TradingContext(**config) as tc:
        exchange = SimulatedExchange()

        assert exchange.base_instrument == tc.base_instrument