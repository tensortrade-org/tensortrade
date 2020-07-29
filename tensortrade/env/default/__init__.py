
import tensortrade.env.default.actions as actions
import tensortrade.env.default.rewards as rewards
import tensortrade.env.default.informers as monitors
import tensortrade.env.default.observers as observers
import tensortrade.env.default.renderers as renderers
import tensortrade.env.default.stoppers as stoppers

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
        renderer_feed=kwargs.get("renderer_feed", None),
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
        stopper=kwargs.get("stopper", stopper),
        informer=kwargs.get("informer", monitors.TensorTradeInformer()),
        renderer=kwargs.get("renderer", renderers.EmptyRenderer())
    )
    return env
