
import pytest
import numpy as np
import pandas as pd
from collections import OrderedDict

import tensortrade.env.default.rewards as rewards

from tensortrade.core import TradingContext
from tensortrade.oms.wallets import Portfolio
from tensortrade.oms.instruments import USD


class TestTensorTradeRewardScheme:

    def setup(self):
        self.config = {
            'base_instrument': 'USD',
            'instruments': 'ETH',
            'rewards': {
                'size': 0
            }
        }

        class ConcreteRewardScheme(rewards.TensorTradeRewardScheme):

            def get_reward(self, env) -> float:
                pass

        self.concrete_class = ConcreteRewardScheme

    def test_injects_reward_scheme_with_context(self):

        with TradingContext(self.config):

            reward_scheme = self.concrete_class()

            assert hasattr(reward_scheme.context, 'size')
            assert reward_scheme.context.size == 0
            assert reward_scheme.context['size'] == 0

    def test_injects_string_initialized_reward_scheme(self):

        with TradingContext(self.config):

            reward_scheme = rewards.get('simple')

            assert reward_scheme.registered_name == "rewards"
            assert hasattr(reward_scheme.context, 'size')
            assert reward_scheme.context.size == 0
            assert reward_scheme.context['size'] == 0


@pytest.fixture
def net_worths():
    return pd.Series([100, 400, 350, 450, 200, 400, 330, 560], name="net_worth")

def net_worths_to_dict(net_worths):
    result = OrderedDict()
    index = 0
    for i in net_worths:
        result[index] = {"net_worth": i}
        index += 1
    return result


def stream_net_worths(net_worths):
    index = 0
    for i in net_worths:
        yield index, {"net_worth": i}
        index += 1


class TestSimpleProfit:

    def test_get_reward(self, net_worths):
        portfolio = Portfolio(USD)
        portfolio._performance = net_worths_to_dict(net_worths)

        pct_chg = net_worths.pct_change()

        reward_scheme = rewards.SimpleProfit()
        assert reward_scheme.get_reward(portfolio) == pct_chg.iloc[-1]  # default window size 1

        reward_scheme._window_size = 3
        reward = ((1 + pct_chg.iloc[-1]) * (1 + pct_chg.iloc[-2]) * (1 + pct_chg.iloc[-3])) - 1
        assert reward_scheme.get_reward(portfolio) == reward


class TestRiskAdjustedReturns:

    def test_sharpe_ratio(self, net_worths):
        scheme = rewards.RiskAdjustedReturns(
            return_algorithm='sharpe',
            risk_free_rate=0,
            window_size=1
        )

        returns = net_worths[-2:].pct_change().dropna()

        expected_ratio = (np.mean(returns) + 1E-9) / (np.std(returns) + 1E-9)
        sharpe_ratio = scheme._sharpe_ratio(returns)

        assert sharpe_ratio == expected_ratio

    def test_sortino_ratio(self, net_worths):
        scheme = rewards.RiskAdjustedReturns(
            return_algorithm='sortino',
            risk_free_rate=0,
            target_returns=0,
            window_size=1
        )

        returns = net_worths[-2:].pct_change().dropna()

        downside_returns = returns.copy()
        downside_returns[returns < 0] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        expected_ratio = (expected_return + 1E-9) / (downside_std + 1E-9)
        sortino_ratio = scheme._sortino_ratio(returns)

        assert sortino_ratio == expected_ratio
