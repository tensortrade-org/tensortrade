import pytest
import numpy as np
from gym.spaces import Discrete
from tensortrade import TradingContext
from tensortrade.actions import MultiDiscreteActions


""" Since the multi-discrete is also dependent on the context it needs to have that. Get that from test_injection.py"""
config = {
    'base_instrument': 'USD',
    'instruments': 'ETH',
    'actions': {
        'n_actions': 50
    }
}



@pytest.fixture()
def context():
    return TradingContext.get_context()


@pytest.fixture(scope='module')
def get_act():
    return MultiDiscreteActions()


@pytest.fixture(scope='module')
def get_act_multi(context):
    """Get the action space for multiple instruments"""

def test_action_is_discrete(get_act):
    """ Test that the action_space is still discrete """
    assert type(get_act.action_space) == Discrete


def test_injects_instruments_into_multi_discrete_scheme():

    with TradingContext(**config) as tc:
        action_scheme = MultiDiscreteActions(actions_per_instrument=25)

        assert action_scheme._instruments == tc.shared['instruments']