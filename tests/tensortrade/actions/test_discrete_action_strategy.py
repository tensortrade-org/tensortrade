
from tensortrade import TradingContext
from tensortrade.actions import DiscreteActionStrategy


c1 = {
        'base_instrument': 'USD',
        'products': ['BTC', 'ETH'],
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}

c2 = {
    'base_instrument': 'USD',
    'products': 'ETH',
    'credentials': {
        'api_key': '48hg34wydghi7ef',
        'api_secret_key': '0984hgoe8d7htg'
    }
}


def test_injects_products_into_strategy():

    with TradingContext(**c1):
        action_strategy = DiscreteActionStrategy(n_actions=25)

    assert action_strategy._product == 'BTC'

    with TradingContext(**c2):
        action_strategy = DiscreteActionStrategy(n_actions=25)

    assert action_strategy._product == 'ETH'

