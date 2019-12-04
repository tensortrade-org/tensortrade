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
    assert isinstance(get_act.action_space.high, np.ndarray)
    assert isinstance(get_act.action_space.low, np.ndarray)


def test_correct_bounds(get_act):
    """ We test that the action_space is within the correct bounds [high: 1, low: 1]"""
    assert get_act.action_space.high[0][0] == 1.0
    assert get_act.action_space.low[0][0] == 0.0

def test_is_correct_sample(get_act):
    """ We test that the action_space is returning the correct output. [[float(0-1)]]"""
    sample = get_act.action_space.sample()
    sample_filtered = sample[0][0]
    assert 0 <= sample_filtered <= 1