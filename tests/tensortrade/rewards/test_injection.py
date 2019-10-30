from tensortrade import TradingContext
from tensortrade.rewards import *
from tensortrade.trades import Trade


class ConcreteRewardStrategy(RewardStrategy):

    def get_reward(self, current_step: int, trade: Trade) -> float:
        pass


config = {
        'base_instrument': 'USD',
        'products': 'ETH',
        'rewards': {
            'amount': 0
        }
}


def test_injects_reward_strategy_with_context():

    with TradingContext(**config) as tc:

        reward_strategy = ConcreteRewardStrategy()

        assert hasattr(reward_strategy.context, 'amount')
        assert reward_strategy.context.amount == 0
        assert reward_strategy.context['amount'] == 0


def test_injects_string_intialized_reward_strategy():

    with TradingContext(**config) as tc:

        reward_strategy = get('simple')

        assert reward_strategy.registered_name == "rewards"
        assert hasattr(reward_strategy.context, 'amount')
        assert reward_strategy.context.amount == 0
        assert reward_strategy.context['amount'] == 0