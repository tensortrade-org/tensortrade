import pytest
import numpy as np
import pandas as pd


from tensortrade.actions import TargetStopActionStrategy
from tensortrade.trades import Trade, TradeType
import tensortrade.exchanges as exchanges


@pytest.fixture
def exchange():
    return exchanges.get('fbm')


class TestTargetStopActionStrategy:

    def test_get_trade_action(self):

        """Testing action calculation"""
        strategy = TargetStopActionStrategy(position_size=20)
        trade_result = strategy.get_trade(action=2, profit_target=50, stop_target=50)
        expected_type = TradeType.MARKET_BUY
        assert expected_type == trade_result[1]

        strategy = TargetStopActionStrategy(position_size=20)
        trade_result = strategy.get_trade(action=7, profit_target=50, stop_target=50)
        expected_type = TradeType.MARKET_BUY
        assert expected_type == trade_result[1]

        """Testing percentage calculation"""
        strategy = TargetStopActionStrategy(position_size=20)
        strategy.get_trade(action=2, profit_target=40, stop_target=20)
        expected_profit = 0.4
        expected_stop_loss = -0.2
        history = strategy.trading_history
        for result in history:
            assert expected_profit == result[3] and expected_stop_loss == result[4]

        """Testing prevention of input above range maximum"""
        strategy = TargetStopActionStrategy(position_size=20, profit_target_range=range(0, 101, 5),
                                            stop_loss_range=range(0, 101, 5))
        strategy.get_trade(action=2, profit_target=9999, stop_target=9999)
        expected_profit = max(strategy.profit_target_range)
        expected_stop_loss = max(strategy.stop_loss_range)
        history = strategy.trading_history
        for result in history:
            assert expected_profit == result[3] and expected_stop_loss == result[4]

        """Testing prevention of input below zero, as well as below range minimum"""
        strategy = TargetStopActionStrategy(position_size=20, profit_target_range=range(0, 101, 5),
                                            stop_loss_range=range(20, 101, 5))
        strategy.get_trade(action=2, profit_target=-500, stop_target=-500)
        expected_profit = min(strategy.profit_target_range)
        expected_stop_loss = min(strategy.stop_loss_range)
        history = strategy.trading_history
        for result in history:
            assert expected_profit == result[3] and expected_stop_loss == result[4]

