from gym.spaces import Discrete

from tensortrade import TradingContext
from tensortrade.actions import *
from tensortrade.trades import Trade


class ConcreteActionScheme(ActionScheme):

    def __init__(self):
        super(ConcreteActionScheme, self).__init__(Discrete(20))

        self.instruments = self.context.instruments

    def get_trade(self, action: TradeActionUnion) -> Trade:
        pass


config = {
    'base_instrument': 'USD',
    'instruments': 'ETH',
    'actions': {
        'n_actions': 50
    }
}


c1 = {
    'base_instrument': 'USD',
    'instruments': ['BTC', 'ETH'],
}

c2 = {
    'base_instrument': 'USD',
    'instruments': 'ETH'
}


def test_injects_strategy():

    with TradingContext(**config):
        action_scheme = ConcreteActionScheme()

        assert hasattr(action_scheme.context, 'n_actions')
        assert action_scheme.context.n_actions == 50
        assert action_scheme.context['n_actions'] == 50


def test_injects_instruments_into_discrete_scheme():

    with TradingContext(**c1):
        action_scheme = DiscreteActions(n_actions=25)

    assert action_scheme._instrument == 'BTC'

    with TradingContext(**c2):
        action_scheme = DiscreteActions(n_actions=25)

    assert action_scheme._instrument == 'ETH'


def test_injects_instruments_into_continuous_scheme():

    with TradingContext(**c1):
        action_scheme = ContinuousActions()

        assert action_scheme._instrument == 'BTC'

    with TradingContext(**c2):
        action_scheme = ContinuousActions()

        assert action_scheme._instrument == 'ETH'


def test_injects_instruments_into_multi_discrete_scheme():

    with TradingContext(**config) as tc:
        action_scheme = MultiDiscreteActions(actions_per_instrument=25)

        assert action_scheme._instruments == tc.shared['instruments']


def test_injects_string_initialized_action_scheme():

    with TradingContext(**config):
        action_scheme = get('discrete')

        assert hasattr(action_scheme.context, 'n_actions')
        assert action_scheme.context.n_actions == 50
        assert action_scheme.context['n_actions'] == 50


def test_injects_discrete_initialization():

    c = {
        'base_instrument': 'USD',
        'instruments': ['BTC', 'ETH'],
        'actions': {
            'n_actions': 25,
            'max_allowed_slippage_percent': 2.0
        }
    }

    with TradingContext(**c):

        action_scheme = DiscreteActions()

        assert action_scheme.n_actions == 25
        assert action_scheme.max_allowed_slippage_percent == 2.0


def test_injects_continuous_initialization():

    c = {
        'base_instrument': 'USD',
        'instruments': ['BTC', 'ETH'],
        'actions': {
            'max_allowed_slippage_percent': 2.0
        }
    }

    with TradingContext(**c):

        action_scheme = ContinuousActions()

        assert action_scheme.max_allowed_slippage_percent == 2.0


def test_injects_multi_discrete_initialization():

    c = {
        'base_instrument': 'USD',
        'instruments': ['BTC', 'ETH'],
        'actions': {
            'actions_per_instrument': 50,
            'max_allowed_slippage_percent': 2.0,
        }
    }

    with TradingContext(**c):

        action_scheme = MultiDiscreteActions()

        assert action_scheme._actions_per_instrument == 50
        assert action_scheme._max_allowed_slippage_percent == 2.0
