import pandas as pd

from gym import Space
from typing import Generator, List, Dict

import tensortrade.slippage as slippage

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
        pass

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


def test_injects_exchange_with_credentials():

    with TradingContext(**config):
        exchange = ConcreteExchange()

        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}
        assert exchange.context['credentials'] == {
            'api_key': '48hg34wydghi7ef', 'api_secret_key': '0984hgoe8d7htg'}


def test_injects_base_instrument():

    with TradingContext(**config):
        exchange = SimulatedExchange()

        assert exchange.base_instrument == 'EURO'


def test_injects_string_initialized_action_scheme():

    with TradingContext(**config):

        exchange = get('simulated')

        assert hasattr(exchange.context, 'credentials')
        assert exchange.context.credentials == config['exchanges']['credentials']
        assert exchange.context['credentials'] == config['exchanges']['credentials']


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


def test_simlulated_from_config():

    class NoSlippage(SlippageModel):

        def fill_order(self, trade: Trade, **kwargs) -> Trade:
            return trade

    config = {
        'base_instrument': 'EURO',
        'instruments': ['BTC', 'ETH'],
        'exchanges': {
            'commission_percent': 0.5,
            'base_precision': 0.3,
            'instrument_precision': 10,
            'min_trade_price': 1e-7,
            'max_trade_price': 1e7,
            'min_trade_size': 1e-4,
            'max_trade_size': 1e4,
            'initial_balance': 1e5,
            'window_size': 5,
            'should_pretransform_obs': True,
            'max_allowed_slippage_percent': 3.0,
            'slippage_model': NoSlippage
        }
    }

    with TradingContext(**config):

        exchange = SimulatedExchange()

        exchange.base_instrument == 'EURO'
        exchange._commission_percent == 0.5
