
from typing import Union

from . import actions
from . import rewards
from . import observers
from . import stoppers
from . import informers
from . import renderers

from tensortrade.env.generic import TradingEnv
from tensortrade.env.generic.components.renderer import AggregateRenderer
from tensortrade.feed.core import DataFeed
from tensortrade.oms.wallets import Portfolio


def create(portfolio: 'Portfolio',
           action_scheme: 'Union[actions.TensorTradeActionScheme, str]',
           reward_scheme: 'Union[rewards.TensorTradeRewardScheme, str]',
           feed: 'DataFeed',
           window_size: int = 1,
           min_periods: int = None,
           **kwargs) -> TradingEnv:
    """Creates the default `TradingEnv` of the project to be used in training
    RL agents.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to be used by the environment.
    action_scheme : `actions.TensorTradeActionScheme` or str
        The action scheme for computing actions at every step of an episode.
    reward_scheme : `rewards.TensorTradeRewardScheme` or str
        The reward scheme for computing rewards at every step of an episode.
    feed : `DataFeed`
        The feed for generating observations to be used in the look back
        window.
    window_size : int
        The size of the look back window to use for the observation space.
    min_periods : int, optional
        The minimum number of steps to warm up the `feed`.
    **kwargs : keyword arguments
        Extra keyword arguments needed to build the environment.

    Returns
    -------
    `TradingEnv`
        The default trading environment.
    """

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

    renderer_list = kwargs.get("renderer", renderers.EmptyRenderer())

    if isinstance(renderer_list, list):
        for i, r in enumerate(renderer_list):
            if isinstance(r, str):
                renderer_list[i] = renderers.get(r)
        renderer = AggregateRenderer(renderer_list)
    else:
        if isinstance(renderer_list, str):
            renderer = renderers.get(renderer_list)
        else:
            renderer = renderer_list

    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=kwargs.get("stopper", stopper),
        informer=kwargs.get("informer", informers.TensorTradeInformer()),
        renderer=renderer
    )
    return env
