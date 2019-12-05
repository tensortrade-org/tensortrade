import pytest
import pandas as pd

from tensortrade.rewards import SimpleProfit

from tensortrade.trades import TradeType, Trade


class TestSimpleProfit:

    def test_raise_reward(self):
        scheme = SimpleProfit()
        scheme.reset()

        # Create the first trade
        trade_1 = Trade(0, "BTC", TradeType.LIMIT_BUY, 100, 1000)
        trade_2 = Trade(2, "BTC", TradeType.LIMIT_SELL, 100, 1500)

        scheme.get_reward(0, trade_1)
        reward1 = scheme.get_reward(2, trade_2)
        
        assert reward1 == 1926.0370135765718
    
    def test_reward_drop(self):
        scheme = SimpleProfit()
        scheme.reset()

        # Create the first trade
        trade_1 = Trade(0, "BTC", TradeType.LIMIT_BUY, 100, 1500)
        trade_2 = Trade(2, "BTC", TradeType.LIMIT_SELL, 100, 1300)

        scheme.get_reward(0, trade_1)
        reward1 = scheme.get_reward(2, trade_2)
        
        assert reward1 == -1015.5908812273915
    
    @pytest.mark.xfail
    def test_reward_invalid_trade_type_input(self):
        scheme = SimpleProfit()
        scheme.reset()

        # Create the first trade
        trade_1 = Trade(0, "BTC", 1, 100, 1500)
        trade_2 = Trade(2, "BTC", TradeType.LIMIT_SELL, 100, 1300)

        scheme.get_reward(0, trade_1)
        reward1 = scheme.get_reward(2, trade_2)
        
        assert reward1 == -1015.5908812273915