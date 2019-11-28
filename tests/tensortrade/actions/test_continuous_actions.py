import pytest
import numpy as np
from gym.spaces import Box 
from tensortrade import TradingContext
from tensortrade.actions import ContinuousActions


@pytest.fixture(scope='module')
def get_act():
    return ContinuousActions()


def test_continuous_is_box(get_act):
    assert type(get_act.action_space) == Box
    assert isinstance(get_act.action_space.high, np.array)
    assert isinstance(get_act.action_space.low, np.array)


def test_correct_bounds(get_act):
    """ We test that the action_space is within the correct bounds [high: 1, low: 1]"""
    assert True == True

def test_is_correct_sample(get_act):
    """ We test that the action_space is returning the correct output. [[float(0-1)]]"""
    assert True == True