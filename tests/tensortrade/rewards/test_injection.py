from tensortrade import TradingContext
from tensortrade.rewards import *
from tensortrade.trades import Trade


class ConcreteRewardScheme(RewardScheme):

    def get_reward(self, current_step: int, trade: Trade) -> float:
        pass


config = {
    'base_instrument': 'USD',
    'instruments': 'ETH',
    'rewards': {
        'amount': 0
    }
}


def test_injects_reward_scheme_with_context():

    with TradingContext(**config):

        reward_scheme = ConcreteRewardScheme()

        assert hasattr(reward_scheme.context, 'amount')
        assert reward_scheme.context.amount == 0
        assert reward_scheme.context['amount'] == 0


def test_injects_string_intialized_reward_scheme():

    with TradingContext(**config):

        reward_scheme = get('simple')

        assert reward_scheme.registered_name == "rewards"
        assert hasattr(reward_scheme.context, 'amount')
        assert reward_scheme.context.amount == 0
        assert reward_scheme.context['amount'] == 0
