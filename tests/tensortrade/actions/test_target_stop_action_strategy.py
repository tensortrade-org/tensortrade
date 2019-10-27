import pytest
import numpy as np
import pandas as pd

from tensortrade.actions import TargetStopActionStrategy
from tensortrade.trades import Trade, TradeType


class TestTargetStopActionStrategy:

    def test_get_trade_profit(self):
        actions = [2, 7, 3, 4, 16]
        for chosen_action in actions:
            strategy = TargetStopActionStrategy(position_size=20, profit_target=1.0,
                                                stop_loss=1.0, trading_history=['BTC', 1, 5])

            get_trade = strategy.get_trade(action=chosen_action, test=True, set_price=1.02)

            expected_trade = Trade('BTC', TradeType.MARKET_SELL, 5, 1.02)

            assert get_trade == expected_trade

    def test_get_trade_(self):
        actions = [2, 7, 3, 4, 16]
        for chosen_action in actions:
            strategy = TargetStopActionStrategy(position_size=20, profit_target=1.0,
                                                stop_loss=1.0, trading_history=['BTC', 1, 5])

            get_trade = strategy.get_trade(action=chosen_action, test=True, set_price=0.98)

            expected_trade = Trade('BTC', TradeType.MARKET_SELL, 5, 1.02)

            assert get_trade == expected_trade
