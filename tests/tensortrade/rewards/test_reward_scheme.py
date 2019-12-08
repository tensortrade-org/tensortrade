
from tensortrade import TradingContext
from tensortrade.exchanges import Exchange
from tensortrade.rewards import RewardScheme

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


def test_exchange_setup():
    with TradingContext(**config):
        reward_scheme = RewardScheme()
        exchange = Exchange(0.0)
        reward_scheme.exchange = exchange

        assert reward_scheme.exchange is exchange


def test_reset_state_scheme():
    reward_scheme = RewardScheme()
    assert reward_scheme.reset() is None
