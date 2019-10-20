from typing import Generator, List, Dict

import pandas as pd
from gym import Space

from tensortrade import TradingContext
from tensortrade.exchanges import InstrumentExchange
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
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_exchange_with_base_instrument():

    with TradingContext(**config) as tc:
        exchange = ConcreteInstrumentExchange()

        assert exchange.context == tc