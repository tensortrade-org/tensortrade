
import ccxt

from tensortrade import TradingContext
from tensortrade.exchanges.live import CCXTExchange


config = {
        'base_instrument': 'USD',
        'products': 'ETH',
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_ccxt_with_credentials():
    exchange = getattr(ccxt, 'coinbase')()

    with TradingContext(**config) as tc:
        exchange = CCXTExchange(exchange)

        assert exchange.base_instrument == config['base_instrument']
        assert exchange._credentials == config['credentials']