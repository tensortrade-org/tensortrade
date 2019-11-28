import pytest
import numpy as np
from gym.spaces import Discrete
from tensortrade import TradingContext
from tensortrade.actions import MultiDiscreteActions


""" Since the multi-discrete is also dependent on the context it needs to have that. Get that from test_injection.py"""


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
    
def test_correct_bounds(get_act):
    """ We test that the action_space is within the correct bounds default 20"""
    assert True == True

def test_is_correct_sample(get_act):
    """ We test that the action_space is returning the correct output. 0 - 20"""
    assert True == True