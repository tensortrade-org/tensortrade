# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np

from abc import abstractmethod
from typing import Dict, Union, List
from gym.spaces import Box

from tensortrade.base import Component, Identifiable
from tensortrade.orders import Order, Broker
from tensortrade.wallets import Portfolio
from tensortrade.instruments import Instrument
from tensortrade.features import FeaturePipeline


class Exchange(Component, Identifiable):
    """An abstract exchange for use within a trading environments.

    Arguments:
        portfolio: The `Portfolio` of tradeable instruments for use on the exchange.
        feature_pipeline (optional): A pipeline of feature transformations for transforming observations.
        kwargs (optional): Optional arguments to augment the functionality of the exchange.
    """
    registered_name = "exchanges"

    def __init__(self, portfolio: Portfolio = None, broker: Broker = None, feature_pipeline: FeaturePipeline = None, **kwargs):
        self._portfolio = self.default('portfolio', portfolio)
        self._broker = self.default('broker', broker) or Broker(self)
        self._feature_pipeline = self.default('feature_pipeline', feature_pipeline)

        self._dtype = self.default('dtype', np.float32, kwargs)
        self._window_size = self.default('window_size', 1, kwargs)
        self._observation_space_lows = self.default('observation_space_lows', None, kwargs)
        self._observation_space_highs = self.default('observation_space_highs', None, kwargs)
        self._observe_wallets = self.default('observe_wallets', None, kwargs)

        if isinstance(self._observe_wallets, list):
            self._observe_unlocked_balances = self._observe_wallets
            self._observe_locked_balances = self._observe_wallets
        else:
            self._observe_unlocked_balances = self.default('observe_unlocked_balances', [], kwargs)
            self._observe_locked_balances = self.default('observe_locked_balances', [], kwargs)

        if not isinstance(self._observe_unlocked_balances, list) or not all(isinstance(balance, Instrument) for balance in self._observe_unlocked_balances):
            raise ValueError(
                'If used, the `self._observe_wallets` or `self._observe_unlocked_balances` parameter must be of type: List[Instrument]')

        if not isinstance(self._observe_locked_balances, list) or not all(isinstance(balance, Instrument) for balance in self._observe_locked_balances):
            raise ValueError(
                'If used, the `self._observe_wallets` or `self._observe_locked_balances` parameter must be of type: List[Instrument]')

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """A pipeline of feature transformations for transforming observations."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self._feature_pipeline = feature_pipeline

    @property
    def portfolio(self) -> Portfolio:
        """The portfolio of instruments currently held on this exchange."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio: Portfolio):
        self._portfolio = portfolio

    @property
    def broker(self) -> Broker:
        """The broker used to execute orders on the exchange."""
        return self._broker

    @broker.setter
    def broker(self, broker: Broker):
        self._broker = broker

    @property
    def window_size(self) -> int:
        """The length of the observation window in the `observation_space`."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

    @property
    def dtype(self) -> Union[type, str]:
        """A type or str corresponding to the dtype of the `observation_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Union[type, str]):
        self._dtype = dtype

    @property
    def wallet_columns(self) -> List[str]:
        """The list of wallet columns provided by the portfolio."""
        unlocked_columns = [instrument.symbol for instrument in self._observe_unlocked_balances]
        locked_columns = ['{}_locked'.format(instrument.symbol)
                          for instrument in self._observe_locked_balances]
        return unlocked_columns + locked_columns

    @property
    def observation_columns(self) -> List[str]:
        """The final list of columns in the observation space."""
        if not self.wallet_columns:
            return self.generated_columns

        return self.generated_columns + self.wallet_columns

    @property
    def observation_space(self) -> Box:
        """The final shape of the observations generated by the exchange each timestep, after any feature transformations."""
        n_features = len(self.observation_columns)

        if self._observation_space_lows and len(self._observation_space_lows) != n_features:
            raise ValueError(
                'The length of `observation_space_lows` provided to the exchange must match the length of `observation_columns`.')

        if self._observation_space_highs and len(self._observation_space_highs) != n_features:
            raise ValueError(
                'The length of `observation_space_highs` provided to the exchange must match the length of `observation_columns`.')

        low = self._observation_space_lows or np.tile(0, n_features)
        high = self._observation_space_highs or np.tile(1e8, n_features)

        if self._window_size > 1:
            low = np.tile(low, self._window_size).reshape((self._window_size, n_features))
            high = np.tile(high, self._window_size).reshape((self._window_size, n_features))

        return Box(low=low, high=high, dtype=self._dtype)

    @property
    def trades(self) -> Dict[str, 'Trade']:
        """A dictionary of trades made on the exchange since the last reset, organized by order id."""
        return self._broker.trades

    @property
    def performance(self) -> pd.DataFrame:
        """The performance of the active account on the exchange since the last reset."""
        return self._portfolio.performance

    def wallet(self, instrument: Instrument) -> 'Wallet':
        wallet = self._portfolio.get_wallet(self.id, instrument)
        return wallet

    def balance(self, instrument: Instrument) -> 'Quantity':
        wallet = self.wallet(instrument=instrument)
        return wallet.balance

    def locked_balance(self, instrument: Instrument) -> 'Quantity':
        wallet = self.wallet(instrument=instrument)
        return wallet.locked_balance

    def observe_balances(self) -> np.ndarray:
        wallets = np.array([])

        for instrument in self._observe_unlocked_balances:
            wallets += [self.balance(instrument).amount]

        for instrument in self._observe_locked_balances:
            wallets += [self.locked_balance(instrument).amount]

        return wallets

    def next_observation(self) -> np.ndarray:
        """Generate the next observation from the exchange, including wallet balances if specified.

        Returns:
            The next multi-dimensional list of observations.
        """
        self._broker.update()
        self._portfolio.update()

        observation = self._generate_next_observation()

        if isinstance(observation, pd.DataFrame):
            observation = observation.fillna(0, axis=1)
            observation = observation.values

        if self._observe_locked_balances or self._observe_unlocked_balances:
            wallet_balances = self.observe_balances()
            observation = observation + wallet_balances

        return observation

    def submit_to_broker(self, order: Order):
        """Submits an order to the exchange's active broker.

        Arguments:
            order: The order to execute.
        """
        return self._broker.submit(order)

    @property
    @abstractmethod
    def generated_columns(self) -> List[str]:
        """The list of generated columns provided by the exchange each timestep, after any feature transformations."""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_next_observation(self) -> bool:
        """If `False`, the exchange's data source has run out of observations.

        Resetting the exchange may be necessary to continue generating observations.

        Returns:
            Whether or not the specified instrument has a next observation.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_next_observation(self) -> Union[pd.DataFrame, np.ndarray]:
        """Utility method to generate the next observation from the exchange,
        including any feature transformations, but excluding wallet balances.

        Returns:
            A `pandas.DataFrame` or `numpy.ndarray` of feature observations.
        """
        raise NotImplementedError()

    @abstractmethod
    def quote_price(self, trading_pair: 'TradingPair') -> float:
        """The quote price of a trading pair on the exchange, denoted in the base instrument.

        Arguments:
            trading_pair: The `TradingPair` to get the quote price for.

        Returns:
            The quote price of the specified trading pair, denoted in the base instrument.
        """
        raise NotImplementedError

    @abstractmethod
    def is_pair_tradeable(self, trading_pair: 'TradingPair') -> bool:
        """Whether or not the specified trading pair is tradeable on this exchange.

        Args:
            trading_pair: The `TradingPair` to test the tradeability of.

        Returns:
            A bool designating whether or not the pair is tradeable.
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_order(self, order: Order):
        """Execute an order on the exchange.

        Arguments:
            order: The order to execute.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the feature pipeline, initial balance, trades, performance, and any other temporary stateful data."""
        if self._portfolio is not None:
            self._portfolio.reset()

        if self._feature_pipeline is not None:
            self.feature_pipeline.reset()

        if self._broker is not None:
            self.broker.reset()
