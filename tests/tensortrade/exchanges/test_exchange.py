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


class ConcreteExchange(Exchange):

    def __init__(self):
        super(ConcreteExchange, self).__init__()

    @property
    def initial_balance(self) -> float:
        pass

    @property
    def balance(self) -> float:
        pass

    @property
    def portfolio(self) -> Dict[str, float]:
        pass

    @property
    def trades(self) -> List[Trade]:
        return []

    @property
    def performance(self) -> pd.DataFrame:
        pass

    @property
    def observation_columns(self) -> List[str]:
        pass

    @property
    def has_next_observation(self) -> bool:
        pass

    def next_observation(self) -> pd.DataFrame:
        pass

    def current_price(self, symbol: str) -> float:
        pass

    def execute_trade(self, trade: Trade) -> Trade:
        pass

    def reset(self):
        pass


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


def test_create_new_exchange(trade_context):
    """ Here we create an entirely new exchange """
    with trade_context:
        exchange = ConcreteExchange()
        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        assert exchange.context['credentials'] == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        
        assert len(exchange.trades) == 0