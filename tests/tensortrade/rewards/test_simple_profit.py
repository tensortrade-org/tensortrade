from unittest import mock

import numpy as np

from tensortrade.rewards import SimpleProfit


def test_reward_not_holding_a_position():
    reward = SimpleProfit()
    trade = mock.Mock()
    trade.configure_mock(is_hold=False, is_buy=False, is_sell=False, amount=0, price=0)
    assert reward.get_reward(0, trade) == -1

    # TODO: Implement all cases for not holding

def test_reward_holding_a_position():
    reward = SimpleProfit()
    trade = mock.Mock()
    # Setup purchase price and hold instrument
    trade.configure_mock(is_hold=False, is_buy=True, is_sell=False, amount=1000, price=100)
    reward.get_reward(0, trade)

    trade.configure_mock(is_hold=True, is_buy=False, is_sell=False, amount=0, price=0)
    assert reward.get_reward(0, trade) == 1

def test_reward_opening_a_position():
    reward = SimpleProfit()
    trade = mock.Mock()
    trade.configure_mock(is_hold=False, is_buy=True, is_sell=False, amount=1000, price=100)
    assert reward.get_reward(0, trade) == 2

def test_reward_closing_a_position():
    reward = SimpleProfit()
    trade = mock.Mock()
    # Setup purchase price and hold instrument
    trade.configure_mock(is_hold=False, is_buy=True, is_sell=False, amount=1000, price=100)
    reward.get_reward(0, trade)

    trade.configure_mock(is_hold=False, is_buy=False, is_sell=True, amount=1000, price=1000)
    profit_per_instrument = 1000 - 100
    profit = 1000 * profit_per_instrument
    profit_sign = np.sign(profit)

    assert reward.get_reward(0, trade) == profit_sign * (1 + (5 ** np.log10(abs(profit))))