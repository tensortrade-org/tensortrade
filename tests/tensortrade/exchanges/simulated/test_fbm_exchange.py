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
from tensortrade.exchanges.simulated import FBMExchange


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


@pytest.fixture(scope="module")
def create_exchange(trade_context):
    with trade_context:
        exchange = FBMExchange()
        exchange.reset()
        return exchange

def test_create_new_exchange(trade_context):
    """ Here we create an entirely new exchange """
    with trade_context:
        exchange = FBMExchange()
        exchange.reset()
        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        assert exchange.context['credentials'] == {
            'api_key': '48hg34wydghi7ef', 
            'api_secret_key': '0984hgoe8d7htg'
        }
        
        assert len(exchange.trades) == 0
        assert exchange._hurst == 0.61
        assert isinstance(exchange.performance, pd.DataFrame)

def test_get_next_observation(create_exchange):
    assert create_exchange.has_next_observation == True
    assert create_exchange._next_observation is not None

def test_get_current_price(create_exchange):
    assert create_exchange.has_next_observation == True
    assert create_exchange.current_price(symbol="ETH")