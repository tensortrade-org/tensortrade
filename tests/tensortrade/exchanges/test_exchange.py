import pandas as pd
import numpy as np

from typing import List

from tensortrade import TradingContext
from tensortrade.data import DataFeed, DataFrameSource
from tensortrade.orders import Trade
from tensortrade.slippage import SlippageModel
from tensortrade.exchanges import Exchange
from tensortrade.instruments import EUR, ETH


class ConcreteExchange(Exchange):

    def __init__(self):
        super(ConcreteExchange, self).__init__()

    @property
    def is_live(self):
        pass

    def quote_price(self, trading_pair: 'TradingPair') -> float:
        pass

    def is_pair_tradable(self, trading_pair: 'TradingPair') -> bool:
        pass

    def execute_order(self, order: 'Order', portfolio: 'Portfolio'):
        pass

    def reset(self):
        pass


config = {
    'base_instrument': EUR,
    'instruments': ETH,
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
        df = pd.Source(
            [[900, 849, 9023, 94039, 943]],
            columns=["open", "high", "low", "close", "volume"]
        )
        exchange = SimulatedExchange(data_frame=df)

        assert exchange._base_instrument == EUR


def test_initialize_ccxt_from_config():

    config = {
        'base_instrument': 'USD',
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


def test_simulated_from_config():

    class NoSlippage(SlippageModel):

        def adjust_trade(self, trade: Trade, **kwargs) -> Trade:
            pass

    config = {
        'base_instrument': 'EURO',
        'instruments': ['BTC', 'ETH'],
        'exchanges': {
            'commission': 0.5,
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
        df = pd.Source(
            [[900, 849, 9023, 94039, 943]],
            columns=["open", "high", "low", "close", "volume"]
        )

        exchange_ds = DataFrameSource('prices', df)
        data_feed = DataFeed([exchange_ds])

        exchange = Exchange('Exchange', lambda x: {EUR/ETH: x['close']})

        assert exchange._base_instrument == 'EURO'
        assert exchange._commission == 0.5


def test_exchange_with_data_source():

    data = np.array([
        [13863.13, 13889., 12952.5, 13480.01, 11484.01],
        [13480.01, 15275., 13005., 14781.51, 23957.87],
        [14781.51, 15400., 14628., 15098.14, 16584.63],
        [15098.14, 15400., 14230., 15144.99, 17980.39],
        [15144.99, 17178., 14824.05, 16960.01, 20781.65]
    ])
    index = pd.Index(
        ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
        name="date"
    )
    columns = ["open", "high", "low", "close", "volume"]
    data_frame = pd.Source(data, index=index, columns=columns)

    data_frame_ds = DataFrameSource('a1', data_frame)
    data_feed = DataFeed([data_frame_ds])
