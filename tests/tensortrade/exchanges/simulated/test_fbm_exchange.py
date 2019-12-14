import pytest
from typing import Generator, List, Dict

import pandas as pd
import tensortrade.slippage as slippage
import copy

from tensortrade import TradingContext
from tensortrade.trades import Trade
from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.trades import TradeType, Trade

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
        assert isinstance(exchange.performance, pd.DataFrame)

def test_get_next_observation(create_exchange):
    assert create_exchange.has_next_observation == True
    assert create_exchange._next_observation is not None


def test_get_current_price(create_exchange):
    """ Test that we're able to get the current price and that it's not 0"""
    assert create_exchange.has_next_observation == True
    assert len(create_exchange.data_frame) != 0
    # This current_price should not be 0 and should not raise and exception.
    assert create_exchange.current_price(symbol="ETH") != 0
    # Check that there are enough price observations

def test_enact_order(create_exchange):
    # Create a trade
    exchange = copy.copy(create_exchange)
    trade_price = exchange.current_price(symbol="ETH")
    trade_1 = Trade(0, "ETH", TradeType.LIMIT_BUY, 100, trade_price)
    exchange._next_observation()
    exchange.execute_trade(trade_1)


    trade_2 = Trade(1, "ETH", TradeType.LIMIT_BUY, 100, trade_price)
    exchange._next_observation()
    exchange.execute_trade(trade_2)
    

    trade_price = exchange.current_price(symbol="ETH")
    trade_3 = Trade(2, "ETH", TradeType.LIMIT_SELL, 73, trade_price)
    exchange._next_observation()
    exchange.execute_trade(trade_3)
    
    
    trade_4 = Trade(3, "ETH", TradeType.LIMIT_SELL, 50, trade_price)
    exchange._next_observation()
    exchange.execute_trade(trade_4)
    

    trade_5 = Trade(4, "ETH", TradeType.LIMIT_SELL, 25, trade_price)
    exchange._next_observation()
    exchange.execute_trade(trade_5)
    
    # Check that we're 5 trades in.
    assert len(exchange.trades) == 5
    assert exchange._current_step == 5

def test_exchange_pretransformed(create_exchange):
    create_exchange.reset()
    assert True == True