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
        self._current_step = 0.0

    @property
    def initial_balance(self) -> float:
        return 0

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    @property
    def trades(self) -> List[Trade]:
        return self._trades

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
        if self._current_step == 0:
            raise AttributeError("This is a random end of file error")

    def current_price(self, symbol: str) -> float:
        pass

    def execute_trade(self, trade: Trade) -> Trade:
        pass

    def reset(self):
        self._current_step = 0
        self._balance = self.initial_balance
        self._portfolio = {}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['step', 'balance', 'net_worth'])


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
def trade_context_resetted(trade_context):
    with trade_context:
        exchange = ConcreteExchange()
        exchange.reset()
        return exchange

def test_create_new_exchange(trade_context):
    """ Here we create an entirely new exchange. There's no logic that sits inside of it, only making sure we are able to set the variables accordingly. """
    with trade_context:
        exchange = ConcreteExchange()
        exchange.reset()
        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        assert exchange.context['credentials'] == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        
        assert len(exchange.trades) == 0
        assert exchange._current_step == 0


@pytest.mark.xfail(raises=AttributeError)
def test_run_reset_error(trade_context_resetted):
    """ Here we check that the module exchange was able to reset. If it reset it should raise an attribute error"""
    trade_context_resetted.reset()
    trade_context_resetted.next_observation