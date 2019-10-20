
from tensortrade import TradingContext
from tensortrade.actions import MultiDiscreteActionStrategy


config = {
        'base_instrument': 'USD',
        'products': ['BTC', 'ETH'],
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_products_into_strategy():

    with TradingContext(**config) as tc:

        action_strategy = MultiDiscreteActionStrategy(actions_per_instrument=25)

    assert action_strategy._products == tc.products
