from typing import Generator, List, Dict

import pandas as pd
from gym import Space

from tensortrade import TradingContext
from tensortrade.exchanges import *
from tensortrade.exchanges.live import *
from tensortrade.exchanges.simulated import *
from tensortrade.trades import Trade


class ConcreteInstrumentExchange(InstrumentExchange):

    def __init__(self):
        super(ConcreteInstrumentExchange, self).__init__()

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
        pass

    @property
    def performance(self) -> pd.DataFrame:
        pass

    @property
    def generated_space(self) -> Space:
        pass

    @property
    def generated_columns(self) -> List[str]:
        pass

    @property
    def has_next_observation(self) -> bool:
        pass

    def _create_observation_generator(self) -> Generator[pd.DataFrame, None, None]:
        pass

    def current_price(self, symbol: str) -> float:
        pass

    def execute_trade(self, trade: Trade) -> Trade:
        pass

    def reset(self):
        pass


config = {
        'base_instrument': 'USD',
        'products': 'ETH',
        'exchanges': {
            'credentials': {
                'api_key': '48hg34wydghi7ef',
                'api_secret_key': '0984hgoe8d7htg'
            }
        }
}


def test_injects_exchange_with_credentials():

    with TradingContext(**config) as tc:
        exchange = ConcreteInstrumentExchange()

        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == {'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        assert exchange.context['credentials'] == {'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}


def test_injects_base_instrument():

    with TradingContext(**config) as tc:
        exchange = SimulatedExchange()

        assert exchange.base_instrument == tc.base_instrument


def test_injects_string_initialized_action_strategy():

    with TradingContext(**config) as tc:

        exchange = get('simulated')

        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == config['exchanges']['credentials']
        assert exchange.context['credentials'] == config['exchanges']['credentials']
