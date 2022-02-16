
from typing import List


import datetime as dt
import numpy as np
import pandas as pd

from gym.spaces import Box, Space
from random import randrange


from tensortrade.feed.core import Stream, NameSpace, DataFeed
from tensortrade.oms.wallets import Wallet
from tensortrade.env.generic import Observer
from collections import OrderedDict


def _create_wallet_source(wallet: 'Wallet', include_worth: bool = True) -> 'List[Stream[float]]':
    """Creates a list of streams to describe a `Wallet`.

    Parameters
    ----------
    wallet : `Wallet`
        The wallet to make streams for.
    include_worth : bool, default True
        Whether or

    Returns
    -------
    `List[Stream[float]]`
        A list of streams to describe the `wallet`.
    """
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    streams = []

    with NameSpace(exchange_name + ":/" + symbol):
        free_balance = Stream.sensor(wallet, lambda w: w.balance.as_float(), dtype="float").rename("free")
        locked_balance = Stream.sensor(wallet, lambda w: w.locked_balance.as_float(), dtype="float").rename("locked")
        total_balance = Stream.sensor(wallet, lambda w: w.total_balance.as_float(), dtype="float").rename("total")

        streams += [free_balance, locked_balance, total_balance]

        if include_worth:
            price = Stream.select(wallet.exchange.streams(), lambda node: node.name.endswith(symbol))
            worth = price.mul(total_balance).rename('worth')
            streams += [worth]

    return streams


def _create_internal_streams(portfolio: 'Portfolio') -> 'List[Stream[float]]':
    """Creates a list of streams to describe a `Portfolio`.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to make the streams for.

    Returns
    -------
    `List[Stream[float]]`
        A list of streams to describe the `portfolio`.
    """
    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += wallet.exchange.streams()
        sources += _create_wallet_source(wallet, include_worth=(symbol != base_symbol))

    worth_streams = []
    for s in sources:
        if s.name.endswith(base_symbol + ":/total") or s.name.endswith("worth"):
            worth_streams += [s]

    net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
    sources += [net_worth]

    return sources


class ObservationHistory(object):
    """Stores observations from a given episode of the environment.

    Parameters
    ----------
    window_size : int
        The amount of observations to keep stored before discarding them.

    Attributes
    ----------
    window_size : int
        The amount of observations to keep stored before discarding them.
    rows : pd.DataFrame
        The rows of observations that are used as the environment observation
        at each step of an episode.

    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.rows = OrderedDict()
        self.index = 0

    def push(self, row: dict) -> None:
        """Stores an observation.

        Parameters
        ----------
        row : dict
            The new observation to store.
        """
        self.rows[self.index] = row
        self.index += 1
        if len(self.rows.keys()) > self.window_size:
            del self.rows[list(self.rows.keys())[0]]

    def observe(self) -> 'np.array':
        """Gets the observation at a given step in an episode

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        rows = self.rows.copy()

        if len(rows) < self.window_size:
            size = self.window_size - len(rows)
            padding = np.zeros((size, len(rows[list(rows.keys())[0]])))
            r = np.array([list(inner_dict.values()) for inner_dict in rows.values()])
            rows = np.concatenate((padding, r))

        if isinstance(rows, OrderedDict):
            rows = np.array([list(inner_dict.values()) for inner_dict in rows.values()])

        rows = np.nan_to_num(rows)

        return rows

    def reset(self) -> None:
        """Resets the observation history"""
        self.rows = OrderedDict()
        self.index = 0


class TensorTradeObserver(Observer):
    """The TensorTrade observer that is compatible with the other `default`
    components.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to be used to create the internal data feed mechanism.
    feed : `DataFeed`
        The feed to be used to collect observations to the observation window.
    renderer_feed : `DataFeed`
        The feed to be used for giving information to the renderer.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    **kwargs : keyword arguments
        Additional keyword arguments for observer creation.

    Attributes
    ----------
    feed : `DataFeed`
        The master feed in charge of streaming the internal, external, and
        renderer data feeds.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    history : `ObservationHistory`
        The observation history.
    renderer_history : `List[dict]`
        The history of the renderer data feed.
    """

    def __init__(self,
                 portfolio: 'Portfolio',
                 feed: 'DataFeed' = None,
                 renderer_feed: 'DataFeed' = None,
                 window_size: int = 1,
                 min_periods: int = None,
                 **kwargs) -> None:
        internal_group = Stream.group(_create_internal_streams(portfolio)).rename("internal")
        external_group = Stream.group(feed.inputs).rename("external")

        if renderer_feed:
            renderer_group = Stream.group(renderer_feed.inputs).rename("renderer")

            self.feed = DataFeed([
                internal_group,
                external_group,
                renderer_group
            ])
        else:
            self.feed = DataFeed([
                internal_group,
                external_group
            ])

        self.window_size = window_size
        self.min_periods = min_periods

        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.inf)
        self._observation_highs = kwargs.get('observation_highs', np.inf)

        self.history = ObservationHistory(window_size=window_size)

        initial_obs = self.feed.next()["external"]
        n_features = len(initial_obs.keys())

        self._observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )

        self.feed = self.feed.attach(portfolio)

        self.renderer_history = []

        self.feed.reset()
        self.warmup()

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def warmup(self) -> None:
        """Warms up the data feed.
        """
        if self.min_periods is not None:
            for _ in range(self.min_periods):
                if self.has_next():
                    obs_row = self.feed.next()["external"]
                    self.history.push(obs_row)

    def observe(self, env: 'TradingEnv') -> np.array:
        """Observes the environment.

        As a consequence of observing the `env`, a new observation is generated
        from the `feed` and stored in the observation history.

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        data = self.feed.next()

        # Save renderer information to history
        if "renderer" in data.keys():
            self.renderer_history += [data["renderer"]]

        # Push new observation to observation history
        obs_row = data["external"]
        self.history.push(obs_row)

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype)
        return obs

    def has_next(self) -> bool:
        """Checks if there is another observation to be generated.

        Returns
        -------
        bool
            Whether there is another observation to be generated.
        """
        return self.feed.has_next()

    def reset(self) -> None:
        """Resets the observer"""
        self.renderer_history = []
        self.history.reset()
        self.feed.reset()
        self.warmup()


class IntradayObserver(Observer):
    """The IntradayObserver observer that is compatible with the other `default`
    components.
    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to be used to create the internal data feed mechanism.
    feed : `DataFeed`
        The feed to be used to collect observations to the observation window.
    renderer_feed : `DataFeed`
        The feed to be used for giving information to the renderer.
    stop_time : datetime.time
        The time at which the episode will stop.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    randomize : bool
        Whether or not to select a random episode when reset.
    **kwargs : keyword arguments
        Additional keyword arguments for observer creation.
    Attributes
    ----------
    feed : `DataFeed`
        The master feed in charge of streaming the internal, external, and
        renderer data feeds.
    stop_time : datetime.time
        The time at which the episode will stop.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    randomize : bool
        Whether or not a random episode is selected when reset.
    history : `ObservationHistory`
        The observation history.
    renderer_history : `List[dict]`
        The history of the renderer data feed.
    """

    def __init__(self,
                 portfolio: 'Portfolio',
                 feed: 'DataFeed' = None,
                 renderer_feed: 'DataFeed' = None,
                 stop_time: 'datetime.time' = dt.time(16, 0, 0),
                 window_size: int = 1,
                 min_periods: int = None,
                 randomize: bool = False,
                 **kwargs) -> None:
        internal_group = Stream.group(_create_internal_streams(portfolio)).rename("internal")
        external_group = Stream.group(feed.inputs).rename("external")

        if renderer_feed:
            renderer_group = Stream.group(renderer_feed.inputs).rename("renderer")

            self.feed = DataFeed([
                internal_group,
                external_group,
                renderer_group
            ])
        else:
            self.feed = DataFeed([
                internal_group,
                external_group
            ])

        self.stop_time = stop_time
        self.window_size = window_size
        self.min_periods = min_periods
        self.randomize = randomize

        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.inf)
        self._observation_highs = kwargs.get('observation_highs', np.inf)

        self.history = ObservationHistory(window_size=window_size)

        initial_obs = self.feed.next()["external"]
        initial_obs.pop('timestamp', None)
        n_features = len(initial_obs.keys())

        self._observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )

        self.feed = self.feed.attach(portfolio)

        self.renderer_history = []

        if self.randomize:
            self.num_episodes = 0
            while self.feed.has_next():
                ts = self.feed.next()["external"]["timestamp"]
                if ts.time() == self.stop_time:
                    self.num_episodes += 1

        self.feed.reset()
        self.warmup()

        self.stop = False

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def warmup(self) -> None:
        """Warms up the data feed.
        """
        if self.min_periods is not None:
            for _ in range(self.min_periods):
                if self.has_next():
                    obs_row = self.feed.next()["external"]
                    obs_row.pop('timestamp', None)
                    self.history.push(obs_row)

    def observe(self, env: 'TradingEnv') -> np.array:
        """Observes the environment.
        As a consequence of observing the `env`, a new observation is generated
        from the `feed` and stored in the observation history.
        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        data = self.feed.next()

        # Save renderer information to history
        if "renderer" in data.keys():
            self.renderer_history += [data["renderer"]]

        # Push new observation to observation history
        obs_row = data["external"]
        try:
            obs_ts = obs_row.pop('timestamp')
        except KeyError:
            raise KeyError("Include Stream of Timestamps named 'timestamp' in feed")
        self.history.push(obs_row)

        # Check if episode should be stopped
        if obs_ts.time() == self.stop_time:
            self.stop = True

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype)
        return obs

    def has_next(self) -> bool:
        """Checks if there is another observation to be generated.
        Returns
        -------
        bool
            Whether there is another observation to be generated.
        """
        return self.feed.has_next() and not self.stop

    def reset(self) -> None:
        """Resets the observer"""
        self.renderer_history = []
        self.history.reset()

        if self.randomize or not self.feed.has_next():
            self.feed.reset()
            if self.randomize:
                episode_num = 0
                while episode_num < randrange(self.num_episodes):
                    ts = self.feed.next()["external"]["timestamp"]
                    if ts.time() == self.stop_time:
                        episode_num += 1

        self.warmup()

        self.stop = False
