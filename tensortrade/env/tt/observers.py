
import numpy as np
import pandas as pd

from gym.spaces import Box, Space

from tensortrade.feed.core import Stream, NameSpace, DataFeed
from tensortrade.oms.wallets import Wallet

from tensortrade.env.generic import Observer


def create_wallet_source(wallet: Wallet, include_worth=True):
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
            worth = (price * total_balance).rename("worth")
            streams += [worth]

    return streams


def create_internal_streams(portfolio: 'Portfolio'):

    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += wallet.exchange.streams()
        sources += create_wallet_source(wallet, include_worth=(symbol != base_symbol))

    worth_streams = []
    for s in sources:
        if s.name.endswith(base_symbol + ":/total") or s.name.endswith("worth"):
            worth_streams += [s]

    net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
    sources += [net_worth]

    return sources


class ObservationHistory(object):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.rows = pd.DataFrame()

    def push(self, row: dict):
        """Saves an observation."""
        self.rows = self.rows.append(row, ignore_index=True)

        if len(self.rows) > self.window_size:
            self.rows = self.rows[-self.window_size:]

    def observe(self):
        rows = self.rows.copy()

        if len(rows) < self.window_size:
            size = self.window_size - len(rows)
            padding = np.zeros((size, rows.shape[1]))
            padding = pd.DataFrame(padding, columns=self.rows.columns)
            rows = pd.concat([padding, rows], ignore_index=True, sort=False)

        if isinstance(rows, pd.DataFrame):
            rows = rows.fillna(0, axis=1)
            rows = rows.values

        rows = np.nan_to_num(rows)

        return rows

    def reset(self):
        self.rows = pd.DataFrame()


class TensorTradeObserver(Observer):

    def __init__(self,
                 portfolio,
                 feed=None,
                 window_size=1,
                 min_periods=None,
                 **kwargs):
        internal_group = create_internal_streams(portfolio).rename("internal")
        external_group = Stream.group(feed.inputs).rename("external")

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
        self.feed.reset()
        self.warmup()

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def warmup(self):
        if self.min_periods is not None:
            for _ in range(self.min_periods):
                if self.has_next():
                    obs_row = self.feed.next()["external"]
                    self.history.push(obs_row)

    def observe(self, env):
        obs_row = self.feed.next()["external"]

        self.history.push(obs_row)

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype)
        return obs

    def has_next(self):
        return self.feed.has_next()

    def reset(self):
        self.history.reset()
        self.feed.reset()
        self.warmup()
