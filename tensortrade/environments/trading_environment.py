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

import gym
import logging
import importlib
import pandas as pd
import numpy as np

from copy import deepcopy
from gym.spaces import Box, Discrete
from typing import Union, Tuple, List, Dict

import tensortrade.exchanges as exchanges
import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.features as features
import tensortrade.wallets as wallets

from tensortrade.base.core import TimeIndexed
from tensortrade.actions import ActionScheme
from tensortrade.rewards import RewardScheme
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline
from tensortrade.instruments import Instrument
from tensortrade.orders import Broker, Order
from tensortrade.wallets import Portfolio

if importlib.util.find_spec("matplotlib") is not None:
    from tensortrade.environments.render import MatplotlibTradingChart


class TradingEnvironment(gym.Env, TimeIndexed):
    """A trading environments made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self,
                 portfolio: Union[Portfolio, str],
                 exchange: Union[Exchange, str],
                 action_scheme: Union[ActionScheme, str],
                 reward_scheme: Union[RewardScheme, str],
                 feature_pipeline: Union[FeaturePipeline, str] = None,
                 window_size: int = 1,
                 **kwargs):
        """
        Arguments:
            exchange: The `Exchange` used to feed data from and execute trades within.
            portfolio: The `Portfolio` of wallets used to submit and execute orders from.
            action_scheme:  The component for transforming an action into an `Order` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            feature_pipeline (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """
        super().__init__()

        self.portfolio = portfolio
        self.exchange = exchange
        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.feature_pipeline = feature_pipeline

        self._window_size = window_size
        self._dtype = kwargs.get('dtype', np.float32)

        self._observation_lows = kwargs.get('observation_lows', 0)
        self._observation_highs = kwargs.get('observation_highs', 1)
        self._observe_wallets = kwargs.get('observe_wallets', None)

        if isinstance(self._observe_wallets, list):
            self._observe_unlocked_balances = self._observe_wallets
            self._observe_locked_balances = self._observe_wallets
        else:
            self._observe_unlocked_balances = kwargs.get('observe_unlocked_balances', [])
            self._observe_locked_balances = kwargs.get('observe_locked_balances', [])

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.viewer = None

        self._enable_logger = kwargs.get('enable_logger', True)

        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        self._initial_balances = None

        self.reset()

    @property
    def window_size(self) -> int:
        """The length of the observation window in the `observation_space`."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

    @property
    def portfolio(self) -> Portfolio:
        """The portfolio of instruments currently held on this exchange."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio: Union[Portfolio, str]):
        self._portfolio = wallets.get(portfolio) if isinstance(portfolio, str) else portfolio

    @property
    def exchange(self) -> Exchange:
        """The `Exchange` that will be used to feed data from and execute trades within."""
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: Union[Exchange, str]):
        self._exchange = exchanges.get(exchange) if isinstance(exchange, str) else exchange

        self._broker = Broker(self._exchange)

    @property
    def broker(self) -> Broker:
        """The broker used to execute orders within the environment."""
        return self._broker

    @property
    def episode_trades(self) -> Dict[str, 'Trade']:
        """A dictionary of trades made this episode, organized by order id."""
        return self._broker.trades

    @property
    def action_scheme(self) -> ActionScheme:
        """The component for transforming an action into an `Order` at each time step."""
        return self._action_scheme

    @action_scheme.setter
    def action_scheme(self, action_scheme: Union[ActionScheme, str]):
        self._action_scheme = actions.get(action_scheme) if isinstance(
            action_scheme, str) else action_scheme

    @property
    def reward_scheme(self) -> RewardScheme:
        """The component for determining the reward at each time step."""
        return self._reward_scheme

    @reward_scheme.setter
    def reward_scheme(self, reward_scheme: Union[RewardScheme, str]):
        self._reward_scheme = rewards.get(reward_scheme) if isinstance(
            reward_scheme, str) else reward_scheme

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """The pipeline of feature transformations to pass the observations through at each time step."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: Union[FeaturePipeline, str] = None):
        self._feature_pipeline = features.get(feature_pipeline) if isinstance(
            feature_pipeline, str) else feature_pipeline

    @property
    def wallet_columns(self) -> List[str]:
        """The list of wallet columns provided by the portfolio."""
        if not isinstance(self._observe_unlocked_balances, list) or not all(isinstance(balance, Instrument) for balance in self._observe_unlocked_balances):
            raise ValueError(
                'If used, the `self._observe_wallets` or `self._observe_unlocked_balances` parameter must be of type: List[Instrument]')

        if not isinstance(self._observe_locked_balances, list) or not all(isinstance(balance, Instrument) for balance in self._observe_locked_balances):
            raise ValueError(
                'If used, the `self._observe_wallets` or `self._observe_locked_balances` parameter must be of type: List[Instrument]')

        unlocked_columns = [instrument.symbol for instrument in self._observe_unlocked_balances]
        locked_columns = ['{}_pending'.format(instrument.symbol)
                          for instrument in self._observe_locked_balances]

        return unlocked_columns + locked_columns

    @property
    def observation_columns(self) -> List[str]:
        """The final list of columns in the observation space."""
        if not self.wallet_columns:
            return self._exchange.observation_columns

        return np.concatenate([self._exchange.observation_columns, self.wallet_columns])

    @property
    def action_space(self) -> Discrete:
        return self._action_scheme.action_space

    @property
    def observation_space(self) -> Box:
        """The final shape of the observations generated by the exchange each timestep, after any feature transformations."""
        n_features = len(self.observation_columns)

        if isinstance(self._observation_lows, list) and len(self._observation_lows) != n_features:
            raise ValueError(
                'The length of `observation_lows` provided to the exchange must match the length of `observation_columns`.')

        if isinstance(self._observation_highs, list) and len(self._observation_highs) != n_features:
            raise ValueError(
                'The length of `observation_highs` provided to the exchange must match the length of `observation_columns`.')

        low = self._observation_lows if isinstance(
            self._observation_lows, list) else np.tile(self._observation_lows, n_features)
        high = self._observation_highs if isinstance(
            self._observation_highs, list) else np.tile(self._observation_highs, n_features)

        if self._window_size > 1:
            low = np.tile(low, self._window_size).reshape((self._window_size, n_features))
            high = np.tile(high, self._window_size).reshape((self._window_size, n_features))

        return Box(low=low, high=high, dtype=self._dtype)

    def wallet(self, instrument: Instrument) -> 'Wallet':
        wallet = self._portfolio.get_wallet(self.exchange.id, instrument)
        return wallet

    def balance(self, instrument: Instrument) -> 'Quantity':
        wallet = self.wallet(instrument=instrument)
        return wallet.balance

    def locked_balance(self, instrument: Instrument) -> 'Quantity':
        wallet = self.wallet(instrument=instrument)
        return wallet.locked_balance

    def observe_balances(self) -> pd.DataFrame:
        wallets = pd.DataFrame([], columns=self.wallet_columns)

        for instrument in self._observe_unlocked_balances:
            wallets[instrument.symbol] = [self.balance(instrument).size]

        for instrument in self._observe_locked_balances:
            wallets['{}_pending'.format(instrument.symbol)] = [
                self.locked_balance(instrument).size]

        return wallets

    def _take_action(self, action: int) -> Order:
        """Determines a specific trade to be taken and executes it within the exchange.

        Arguments:
            action: The int provided by the agent to map to a trade action for this timestep.

        Returns:
            The order created by the agent this time step, if any.
        """
        order = self._action_scheme.get_order(action, self._exchange, self._portfolio)

        if order:
            self._broker.submit(order)

        self._broker.update()
        self._portfolio.update()

        return order

    def _next_observation(self) -> np.ndarray:
        """Returns the next observation from the exchange.

        Returns:
            The observation provided by the environments's exchange, often OHLCV or tick trade history data points.
        """
        observation = self._exchange.next_observation()

        if self._observe_locked_balances or self._observe_unlocked_balances:
            wallet_balances = self.observe_balances()

            for column in list(wallet_balances.columns):
                observation.loc[observation.index[0], column] = wallet_balances[column].values

        if self._feature_pipeline is not None:
            observation = self._feature_pipeline.transform(observation)

        if len(observation) < self._window_size:
            padding = np.zeros((self._window_size - len(observation),
                                len(self.observation_columns)))
            padding = pd.DataFrame(padding, columns=self.observation_columns)
            observation = pd.concat([padding, observation], ignore_index=True, sort=False)

        observation = observation.select_dtypes(include='number')

        if isinstance(observation, pd.DataFrame):
            observation = observation.fillna(0, axis=1)
            observation = observation.values

        if len(observation) != 0:
            observation = observation[0]

        observation = np.nan_to_num(observation)

        return observation

    def _get_reward(self) -> float:
        """Returns the reward for the current timestep.

        Returns:
            A float corresponding to the benefit earned by the action taken this step.
        """
        reward = self._reward_scheme.get_reward(self._portfolio)
        reward = np.nan_to_num(reward)

        if np.bitwise_not(np.isfinite(reward)):
            raise ValueError('Reward returned by the reward scheme must by a finite float.')

        return reward

    def _done(self) -> bool:
        """Returns whether or not the environments is done and should be restarted.

        Returns:
            A boolean signaling whether the environments is done and should be restarted.
        """
        lost_90_percent_net_worth = self._portfolio.profit_loss < 0.1
        return lost_90_percent_net_worth or not self._exchange.has_next_observation

    def _info(self, order: Order) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.

        Args:
            order: The order created during the currente timestep.

        Returns:
            info: A dictionary containing the exchange used, the portfolio, the broker,
                  the current timestep, and any order executed this time step.
        """
        return {
            'current_step': self.clock.step,
            'portfolio': self._portfolio,
            'broker': self._broker,
            'exchange': self._exchange,
            'order': order,
        }

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environments based on the specified action.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environments's exchange, often OHLCV or tick trade history data points.
            reward (float): An size corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environments is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """
        order = self._take_action(action)

        observation = self._next_observation()
        reward = self._get_reward()
        done = self._done()
        info = self._info(order)

        if self._enable_logger:
            self.logger.debug('Order: {}'.format(order))
            self.logger.debug('Observation: {}'.format(observation))
            self.logger.debug('P/L: {}'.format(self._portfolio.profit_loss))
            self.logger.debug('Reward ({}): {}'.format(self.clock.step, reward))
            self.logger.debug('Performance: {}'.format(self._portfolio.performance.tail(1)))

        self.clock.increment()

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """Resets the state of the environments and returns an initial observation.

        Returns:
            The episode's initial observation.
        """
        self.clock.reset()

        if not self._exchange.is_live:
            if self._initial_balances is not None:
                self._portfolio._wallets = {}

                for balance in self._initial_balances:
                    self._portfolio.add((self._exchange, balance.instrument, balance.size))
            else:
                self._initial_balances = self._portfolio.total_balances

        self._action_scheme.reset()
        self._reward_scheme.reset()
        self._exchange.reset()
        self._portfolio.reset()
        self._broker.reset()

        observation = self._next_observation()

        self.clock.increment()

        return observation

    def render(self, mode='none'):
        """Renders the environment via matplotlib."""
        if mode == 'log':
            self.logger.info('Performance: ' + str(self._portfolio.performance))
        elif mode == 'chart':
            if self.viewer is None and hasattr(self.exchange, '_pre_transformed_data'):
                self.viewer = MatplotlibTradingChart(self.exchange._pre_transformed_data)

            if self.viewer is not None:
                self.viewer.render(self.clock.step - 1,
                                   self._portfolio.performance['net_worth'].values,
                                   self.render_benchmarks,
                                   self._broker.trades)

    def close(self):
        """Utility method to clean environment before closing."""
        if self.viewer is not None:
            self.viewer.close()
