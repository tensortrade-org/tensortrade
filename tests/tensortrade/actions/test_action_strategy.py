

from gym.spaces import Discrete

from tensortrade import TradingContext
from tensortrade.actions import ActionStrategy, TradeActionUnion
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
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_strategy_with_products():

    with TradingContext(**config) as tc:
        action_strategy = ConcreteActionStrategy()

        assert action_strategy.context == tc
