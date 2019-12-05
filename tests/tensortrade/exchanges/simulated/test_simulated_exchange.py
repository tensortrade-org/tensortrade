import pytest
from typing import Generator, List, Dict

import pandas as pd
import tensortrade.slippage as slippage

from gym import Space

from tensortrade import TradingContext
from tensortrade.trades import Trade
from tensortrade.slippage import SlippageModel
from tensortrade.exchanges import Exchange, get
from tensortrade.exchanges.live import CCXTExchange
from tensortrade.exchanges.simulated import SimulatedExchange, FBMExchange


config = {
    'base_instrument': 'EURO',
    'instruments': 'ETH',
    'exchanges': {
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
    }
}

@pytest.fixture(scope="module")
def trade_context():
    return TradingContext(**config)


def test_create_injected_simulated_exchange(trade_context):

    with trade_context:
        exchange = SimulatedExchange()
        exchange.reset()
        assert exchange.base_instrument == 'EURO'
        assert exchange.initial_balance == 1e4
        assert exchange._current_step == 0

def test_exchange_pretransform_true():
    """ Test what would happen for pretransform == True """
    
    assert True == True

def test_exchange_pretransform_false():
    """ Test what would happen for pretransform == True """
    
    assert True == True