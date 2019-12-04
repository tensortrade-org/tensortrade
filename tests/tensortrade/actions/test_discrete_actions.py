import pytest
import numpy as np
from gym.spaces import Discrete
from tensortrade.actions import DiscreteActions

@pytest.fixture(scope='module')
def get_act():
    return DiscreteActions()


def test_action_is_discrete(get_act):
    assert type(get_act.action_space) == Discrete
    assert get_act.action_space.n == 20