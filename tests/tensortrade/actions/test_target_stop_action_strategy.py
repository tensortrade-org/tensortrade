import pytest
import numpy as np
import pandas as pd

from tensortrade.actions import TargetStopActionStrategy
from tensortrade.trades import Trade, TradeType


class TestTargetStopActionStrategy:

    def test_get_trade_action(self):

        strategy = TargetStopActionStrategy(position_size=20)
        trade_result = strategy.get_trade(action=2)
        expected_type = TradeType.MARKET_BUY
        expected_amount = strategy._exchange.instrument_balance(strategy.instrument_symbol, 0) * 0.25
        assert expected_type == trade_result[1] and expected_amount == trade_result[2]

        strategy = TargetStopActionStrategy(position_size=20)
        strategy.get_trade(action=2)
        expected_profit = 0.75
        expected_stop_loss = -0.75
        history = strategy.trading_history
        assert expected_profit == history[3] and expected_stop_loss == history[4]
