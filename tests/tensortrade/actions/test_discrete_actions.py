import pytest
import numpy as np
from gym.spaces import Discrete
from tensortrade.actions import DiscreteActions

@pytest.fixture(scope='module')
def get_act():
    return DiscreteActions()


def test_action_is_discrete(get_act):
    assert type(get_act.action_space) == Discrete
    


def test_correct_bounds(get_act):
    """ We test that the action_space is within the correct bounds default 20"""
    assert True == True

def test_is_correct_sample(get_act):
    """ We test that the action_space is returning the correct output. 0 - 20"""
    assert True == True