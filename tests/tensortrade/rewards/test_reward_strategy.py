

from tensortrade import TradingContext
from tensortrade.rewards import RewardStrategy
from tensortrade.trades import Trade


class ConcreteRewardStrategy(RewardStrategy):

    def get_reward(self, current_step: int, trade: Trade) -> float:
        pass


config = {
        'base_instrument': 'USD',
        'products': 'ETH',
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_reward_strategy_with_context():

    with TradingContext(**config) as tc:

        reward_strategy = ConcreteRewardStrategy()

        assert reward_strategy.context == tc

