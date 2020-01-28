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
import uuid
import logging
import numpy as np

from gym.spaces import Discrete, Space, Box
from typing import Union, Tuple, List, Dict

import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.wallets as wallets

from tensortrade.base import TimeIndexed, Clock
from tensortrade.actions import ActionScheme
from tensortrade.rewards import RewardScheme
from tensortrade.data import DataFeed, Select
from tensortrade.data.internal import create_internal_feed
from tensortrade.orders import Broker, Order
from tensortrade.wallets import Portfolio
from tensortrade.environments import ObservationHistory


class TradingEnvironment(gym.Env, TimeIndexed):
    """A trading environments made for use with Gym-compatible reinforcement learning algorithms."""

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 portfolio: Union[Portfolio, str],
                 action_scheme: Union[ActionScheme, str],
                 reward_scheme: Union[RewardScheme, str],
                 feed: DataFeed = None,
                 window_size: int = 1,
                 use_internal=True,
                 **kwargs):
        """
        Arguments:
            portfolio: The `Portfolio` of wallets used to submit and execute orders from.
            action_scheme:  The component for transforming an action into an `Order` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            feed (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """
        super().__init__()

        self.portfolio = portfolio
        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.feed = feed
        self.window_size = window_size
        self.use_internal = use_internal

        if self.feed:
            self._external_keys = self.feed.next().keys()
            self.feed.reset()

        self.history = ObservationHistory(window_size=window_size)
        self._broker = Broker(exchanges=self.portfolio.exchanges)

        self.clock = Clock()
        self.action_space = None
        self.observation_space = None
        self.viewer = None

        self._enable_logger = kwargs.get('enable_logger', False)
        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', 0)
        self._observation_highs = kwargs.get('observation_highs', 1)

        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        self.compile()

    def compile(self):
        """
        Sets the observation space and the action space of the environment.
        Creates the internal feed and sets initialization for different components.
        """
        for component in [self._broker, self.portfolio, self.action_scheme, self.reward_scheme]:
            component.clock = self.clock

        self.action_scheme.set_pairs(exchange_pairs=self.portfolio.exchange_pairs)
        self.action_space = Discrete(len(self.action_scheme))

        if not self.feed:
            self.feed = create_internal_feed(self.portfolio)

        self.feed = self.feed + create_internal_feed(self.portfolio)

        initial_obs = self.feed.next()
        n_features = len(initial_obs.keys()) if self.use_internal else len(self._external_keys)

        self.observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )

        self.feed.reset()

    @property
    def portfolio(self) -> Portfolio:
        """The portfolio of instruments currently held on this exchange."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio: Union[Portfolio, str]):
        self._portfolio = wallets.get(portfolio) if isinstance(portfolio, str) else portfolio

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

    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        """Run one timestep within the environments based on the specified action.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environments's exchange, often OHLCV or tick trade history data points.
            reward (float): An size corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environments is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """
        order = self.action_scheme.get_order(action, self.portfolio)

        if order:
            self._broker.submit(order)

        self._broker.update()

        obs_row = self.feed.next()

        if not self.use_internal:
            obs_row = {k: obs_row[k] for k in self._external_keys}

        self.history.push(obs_row)

        obs = self.history.observe()

        reward = self.reward_scheme.get_reward(self._portfolio)
        reward = np.nan_to_num(reward)

        if np.bitwise_not(np.isfinite(reward)):
            raise ValueError('Reward returned by the reward scheme must by a finite float.')

        done = (self.portfolio.profit_loss < 0.1) or not self.feed.has_next()

        info = {
            'step': self.clock.step,
            'portfolio': self.portfolio,
            'broker': self._broker,
            'order': order,
        }

        if self._enable_logger:
            self.logger.debug('Order:       {}'.format(order))
            self.logger.debug('Observation: {}'.format(obs))
            self.logger.debug('P/L:         {}'.format(self._portfolio.profit_loss))
            self.logger.debug('Reward ({}): {}'.format(self.clock.step, reward))
            self.logger.debug('Performance: {}'.format(self._portfolio.performance.tail(1)))

        self.clock.increment()

        return obs, reward, done, info

    def reset(self) -> np.array:
        """Resets the state of the environments and returns an initial observation.

        Returns:
            The episode's initial observation.
        """
        self.episode_id = uuid.uuid4()

        self.clock.reset()
        self.feed.reset()
        self.action_scheme.reset()
        self.reward_scheme.reset()
        self.portfolio.reset()
        self.history.reset()
        self._broker.reset()

        obs_row = self.feed.next()

        if not self.use_internal:
            obs_row = {k: obs_row[k] for k in self._external_keys}

        self.history.push(obs_row)

        obs = self.history.observe()

        self.clock.increment()

        return obs

    def render(self, mode='none'):
        """Renders the environment via matplotlib."""
        if mode == 'log':
            self.logger.info('Performance: ' + str(self._portfolio.performance))
        elif mode == 'chart':
            if self.viewer is None:
                raise NotImplementedError()

            self.viewer.render(self.clock.step - 1,
                               self._portfolio.performance,
                               self._broker.trades)

    def close(self):
        """Utility method to clean environment before closing."""
        if self.viewer is not None:
            self.viewer.close()
