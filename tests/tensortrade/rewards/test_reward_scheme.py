from tensortrade import TradingContext
from tensortrade.rewards import get, RewardScheme
from tensortrade.trades import Trade


class ConcreteRewardScheme(RewardScheme):

    def get_reward(self, current_step: int, trade: Trade) -> float:
        pass


config = {
    'base_instrument': 'USD',
    'instruments': 'ETH',
    'rewards': {
        'size': 0
    }
}


def test_injects_reward_scheme_with_context():

    with TradingContext(**config):

        reward_scheme = ConcreteRewardScheme()

        assert hasattr(reward_scheme.context, 'size')
        assert reward_scheme.context.size == 0
        assert reward_scheme.context['size'] == 0


def test_injects_string_intialized_reward_scheme():

    with TradingContext(**config):

        reward_scheme = get('simple')

        assert reward_scheme.registered_name == "rewards"
        assert hasattr(reward_scheme.context, 'size')
        assert reward_scheme.context.size == 0
        assert reward_scheme.context['size'] == 0
