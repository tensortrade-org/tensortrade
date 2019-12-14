
import ccxt
import pytest
from typing import Generator, List, Dict

import pandas as pd
from gym import Space
from tensortrade.trades import Trade
from tensortrade import TradingContext
from tensortrade.exchanges.live import CCXTExchange


def test_initialize_ccxt_from_config():

    config = {
        'base_instrument': 'USD',
        'instruments': 'ETH',
        'exchanges': {
            'exchange': 'binance',
            'credentials': {
                'api_key': '48hg34wydghi7ef',
                'api_secret_key': '0984hgoe8d7htg'
            }
        }
    }

    with TradingContext(**config):

        exchange = CCXTExchange()

        assert str(exchange._exchange) == 'Binance'
        assert exchange._credentials == config['exchanges']['credentials']
    



@pytest.mark.xfail(raises=ccxt.AuthenticationError)
def test_get_failed_next_observation():
    """ Test that the the authentication isn't valid (Non-valid keys)"""
    config = {
        'base_instrument': 'USD',
        'instruments': 'ETH',
        'exchanges': {
            'exchange': 'binance',
            'credentials': {
                'api_key': '48hg34wydghi7ef',
                'api_secret_key': '0984hgoe8d7htg'
            }
        }
    }

    with TradingContext(**config):

        exchange = CCXTExchange()
        exchange.reset()
        assert str(exchange._exchange) == 'Binance'
        assert exchange._credentials == config['exchanges']['credentials']