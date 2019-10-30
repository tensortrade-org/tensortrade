from gym.spaces import Discrete

from tensortrade import TradingContext
from tensortrade.actions import *
from tensortrade.trades import Trade


class ConcreteActionStrategy(ActionStrategy):

    def __init__(self):
        super(ConcreteActionStrategy, self).__init__(Discrete(20))
        self.products = self.context.products

    def get_trade(self, action: TradeActionUnion) -> Trade:
        pass


config = {
        'base_instrument': 'USD',
        'products': 'ETH',
        'actions': {
            'n_actions': 50
        }
}


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


def test_injects_strategy():

    with TradingContext(**config) as tc:
        action_strategy = ConcreteActionStrategy()

        assert hasattr(action_strategy.context, 'n_actions')
        assert action_strategy.context.n_actions == 50
        assert action_strategy.context['n_actions'] == 50


def test_injects_products_into_discrete_strategy():

    with TradingContext(**c1):
        action_strategy = DiscreteActionStrategy(n_actions=25)

    assert action_strategy._product == 'BTC'

    with TradingContext(**c2):
        action_strategy = DiscreteActionStrategy(n_actions=25)

    assert action_strategy._product == 'ETH'


def test_injects_products_into_continuous_strategy():

    with TradingContext(**c1):
        action_strategy = ContinuousActionStrategy()

    assert action_strategy._product == 'BTC'

    with TradingContext(**c2):
        action_strategy = ContinuousActionStrategy()

    assert action_strategy._product == 'ETH'


def test_injects_products_into_multi_discrete_strategy():

    with TradingContext(**config) as tc:

        action_strategy = MultiDiscreteActionStrategy(actions_per_instrument=25)

    assert action_strategy._products == tc.products


def test_injects_string_initialized_action_strategy():

    with TradingContext(**config) as tc:

        action_strategy = get('discrete')

        assert hasattr(action_strategy.context, 'n_actions')
        assert action_strategy.context.n_actions == 50
        assert action_strategy.context['n_actions'] == 50