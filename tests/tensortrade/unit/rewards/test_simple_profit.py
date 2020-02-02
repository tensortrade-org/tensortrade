import pytest
import pandas as pd

from tensortrade.rewards import SimpleProfit
from tensortrade.wallets import Portfolio
from tensortrade.instruments import USD

@pytest.fixture
def net_worths():
    return pd.Series([100, 400, 350, 450, 200, 400, 330, 560], name="net_worth")

class TestSimpleProfit:

    def test_get_reward(self, net_worths):
        portfolio = Portfolio(USD)
        performance = pd.DataFrame({ 'net_worth': net_worths })
        portfolio._performance = performance

        pct_chg = net_worths.pct_change()

        reward_scheme = SimpleProfit()
        assert reward_scheme.get_reward(portfolio) == pct_chg.iloc[-1]  # default window size 1

        reward_scheme.window_size = 3
        reward = ((1 + pct_chg.iloc[-1]) * (1 + pct_chg.iloc[-2]) * (1 + pct_chg.iloc[-3])) - 1
        assert reward_scheme.get_reward(portfolio) == reward
