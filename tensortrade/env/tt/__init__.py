
import tensortrade.env.tt.actions as actions
import tensortrade.env.tt.rewards as rewards
import tensortrade.env.tt.monitors as monitors
import tensortrade.env.tt.observers as observers
import tensortrade.env.tt.renderers as renderers
import tensortrade.env.tt.stoppers as stoppers

from typing import Union

from tensortrade.env.generic import TradingEnv
from tensortrade.feed.core import DataFeed


def create(portfolio,
           action_scheme: Union[actions.TensorTradeActionScheme, str],
           reward_scheme: Union[rewards.TensorTradeRewardScheme, str],
           feed: DataFeed,
           window_size: int = 1,
           min_periods: int = None,
           **kwargs) -> TradingEnv:

    action_scheme = actions.get(action_scheme) if isinstance(action_scheme, str) else action_scheme
    reward_scheme = rewards.get(reward_scheme) if isinstance(reward_scheme, str) else reward_scheme

    action_scheme.portfolio = portfolio

    observer = observers.TensorTradeObserver(
        portfolio=portfolio,
        feed=feed,
        window_size=window_size,
        min_periods=min_periods
    )

    stopper = stoppers.MaxLossStopper(
        max_allowed_loss=kwargs.get("max_allowed_loss", 0.5)
    )

    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=stopper,
        monitor=monitors.TensorTradeMonitor(),
        renderer=renderers.EmptyRenderer()
    )
    return env
