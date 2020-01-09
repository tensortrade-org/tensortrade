
import pytest
import pandas as pd

from tensortrade.base import TradingContext
from tensortrade.exchanges.simulated import StochasticExchange
from tensortrade.actions import ManagedRiskOrders
from tensortrade.rewards import RiskAdjustedReturns
from tensortrade.features import FeatureTransformer, FeaturePipeline
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.instruments import USD, BTC
from tensortrade.trades import TradeSide, TradeType
from tensortrade.environments import TradingEnvironment
from typing import Union, List


def rsi(X: pd.DataFrame,
        period: int = 14):
    """
    Computes the relative strength index.

    Recommended by Wilder according to Wikipedia page.

    References:
        - https://en.wikipedia.org/wiki/Relative_strength_index

    Arguments:
        X : pd.DataFrame
        period : int = 14
    """
    returns = X.close.diff()

    U = returns.apply(lambda x: x if x > 0 else 0)
    D = returns.apply(lambda x: -x if x < 0 else 0)

    SMMA = lambda s, n: s.ewm(alpha=1 / n).mean()

    RS = SMMA(U, period) / SMMA(D, period)

    RSI = 100 - (100 / (1 + RS))
    RSI = pd.Series(RSI, name="RSI_{}".format(period))

    return RSI


def macd(X : pd.DataFrame,
         lower : int = 7,
         upper : int = 14,
         price_window : int = 63,
         window : int = 252):
    """
    Computes the Moving Average Convergence Divergence.

    References:
        - https://arxiv.org/pdf/1911.10107.pdf

    Arguments:
        X : pd.DataFrame
          A pandas dataframe to be used in the transformation.
        lower : int = 12
          Used for the span of the EWMA of the prices for the returns.
        upper : int = 7,
        price_window : int = 63,
        window : int = 252
    """
    prices = X.close
    ewma = lambda span: prices.ewm(span=span, adjust=False).mean()

    q = (ewma(upper) - ewma(lower)) / prices.rolling(price_window).std()
    macd = q / prices.rolling(window).std()
    macd = pd.Series(macd, name="MACD_({},{})".format(lower, upper))

    return macd


class FunctionsTransformer(FeatureTransformer):

    def __init__(
            self,
            columns: Union[List[str], str, None] = None,
            inplace=True,
            specifications: List[tuple] = None,
            window_size: int = None
    ):
        super().__init__(columns, inplace)
        self.specifications = specifications
        self.window_size = window_size

    def transform(self, X: pd.DataFrame):

        df = pd.DataFrame()
        for f, kwargs in self.specifications:
            series = f(X, **kwargs)
            df[series.name] = series

        if self._inplace:
            return pd.concat([X, df], axis=1)

        return df


@pytest.fixture
def env():
    context = {
        "base_instrument": USD,
        "actions": {
            "pairs": [USD / BTC],
            "stop_loss_percentages": [0.02, 0.04, 0.06],
            "take_profit_percentages": [0.01, 0.02, 0.03],
            "trade_sizes": 10,
            "trade_side": TradeSide.BUY,
            "trade_type": TradeType.MARKET,
            "order_listener": None
        },
        "rewards": {
            "return_algorithm": "sharpe",
            "risk_free_rate": 0,
            "target_returns": 0
        },
        "exchanges": {
            "model_type": "FBM",
            "hurst": 0.61,
            "timeframe": "1d",
            "base_price": 7500,
            "base_volume": 12000
        }
    }

    with TradingContext(**context):
        action_scheme = ManagedRiskOrders()
        reward_scheme = RiskAdjustedReturns()
        exchange = StochasticExchange()

        portfolio = Portfolio(USD, [
            Wallet(exchange, 100000 * USD),
            Wallet(exchange, 0 * BTC)
        ])

        env = TradingEnvironment(
            portfolio=portfolio,
            exchange=exchange,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            window_size=14,
            enable_logger=False
        )

    return env


@pytest.fixture
def env():
    context = {
        "base_instrument": USD,
        "actions": {
            "pairs": [USD / BTC],
            "stop_loss_percentages": [0.02, 0.04, 0.06],
            "take_profit_percentages": [0.01, 0.02, 0.03],
            "trade_sizes": 10,
            "trade_side": TradeSide.BUY,
            "trade_type": TradeType.MARKET,
            "order_listener": None
        },
        "rewards": {
            "return_algorithm": "sharpe",
            "risk_free_rate": 0,
            "target_returns": 0
        },
        "exchanges": {
            "model_type": "FBM",
            "hurst": 0.61,
            "timeframe": "1d",
            "base_price": 7500,
            "base_volume": 12000
        }
    }

    with TradingContext(**context):
        action_scheme = ManagedRiskOrders()
        reward_scheme = RiskAdjustedReturns()
        exchange = StochasticExchange()

        portfolio = Portfolio(USD, [
            Wallet(exchange, 100000 * USD),
            Wallet(exchange, 0 * BTC)
        ])

        env = TradingEnvironment(
            portfolio=portfolio,
            exchange=exchange,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            window_size=14,
            enable_logger=False
        )

    return env


def test_insufficient_funds(env):
    done = False

    total_reward = 0

    while not done:

        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        total_reward += reward

    assert done


