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


import logging
import gym
import numpy as np

from gym.spaces import Discrete
from typing import Union, Tuple, List, Dict

import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.wallets as wallets

from tensortrade.base.core import TimeIndexed, Clock
from tensortrade.actions import ActionScheme
from tensortrade.rewards import RewardScheme
from tensortrade.data import DataFeed
from tensortrade.orders import Broker, Order
from tensortrade.wallets import Portfolio
from tensortrade.environments.history import History


class TradingEnvironment(gym.Env, TimeIndexed):
    """A trading environments made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self,
                 portfolio: Union[Portfolio, str],
                 action_scheme: Union[ActionScheme, str],
                 reward_scheme: Union[RewardScheme, str],
                 feed: DataFeed,
                 window_size: int = 1,
                 history: History = None,
                 **kwargs):
        """
        Arguments:
            portfolio: The `Portfolio` of wallets used to submit and execute orders from.
            action_scheme:  The component for transforming an action into an `Order` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            feature_pipeline (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """
        super().__init__()
        TimeIndexed.clock = Clock()

        self.portfolio = portfolio
        self.action_scheme = action_scheme
        self.action_scheme.over(exchange_pairs=self.portfolio.exchange_pairs)
        self.reward_scheme = reward_scheme
        self.feed = feed
        self.history = history if history else History(window_size=window_size)
        self._broker = Broker(exchanges=self.portfolio.exchanges)

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.viewer = None

        self._enable_logger = kwargs.get('enable_logger', True)

        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

    @property
    def action_space(self):
        return Discrete(len(self.action_scheme))

    @property
    def observation_shape(self):
        obs = self.observation_shape

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

    def _take_action(self, action: int) -> Order:
        """Determines a specific trade to be taken and executes it within the exchange.

        Arguments:
            action: The int provided by the agent to map to a trade action for this timestep.

        Returns:
            The order created by the agent this time step, if any.
        """
        order = self._action_scheme.get_order(action, self.portfolio)

        if order:
            self.broker.submit(order)

        self.broker.update()
        self.portfolio.update()

        return order

    def _next_observation(self) -> np.ndarray:
        """Returns the next observation from the exchange.

        Returns:
            The observation provided by the environments's exchange, often OHLCV or tick trade history data points.
        """

        obs = self.feed.next()

        self.history.push(obs)

        return self.history.observe()

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
        return lost_90_percent_net_worth or not self.feed.has_next()

    def _info(self, order: Order) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.

        Args:
            order: The order created during the current timestep.

        Returns:
            info: A dictionary containing the exchange used, the portfolio, the broker,
                  the current timestep, and any order executed this time step.
        """
        return {
            'current_step': self.clock.step,
            'portfolio': self._portfolio,
            'broker': self._broker,
            'order': order,
        }

    def compile(self):
        obs = self.feed.next()

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
        order = self._take_action(action)

        observation = self._next_observation()
        reward = self._get_reward()
        done = self._done()
        info = self._info(order)

        if self._enable_logger:
            self.logger.debug('Order:       {}'.format(order))
            self.logger.debug('Observation: {}'.format(observation))
            self.logger.debug('P/L:         {}'.format(self._portfolio.profit_loss))
            self.logger.debug('Reward ({}): {}'.format(self.clock.step, reward))
            self.logger.debug('Performance: {}'.format(self._portfolio.performance.tail(1)))

        self.clock.increment()

        return observation, reward, done, info

    def reset(self) -> np.array:
        """Resets the state of the environments and returns an initial observation.

        Returns:
            The episode's initial observation.
        """
        self.clock.reset()
        self.feed.reset()
        self.action_scheme.reset()
        self.reward_scheme.reset()
        self.portfolio.reset()
        self.broker.reset()

        observation = self._next_observation()

        self.clock.increment()

        return observation

    def render(self, mode='none'):
        """Renders the environment via matplotlib."""
        if mode == 'log':
            self.logger.info('Performance: ' + str(self._portfolio.performance))
        elif mode == 'chart':
            if self.viewer is not None:
                self.viewer.render(self.clock.step - 1,
                                   self._portfolio.performance['net_worth'].values,
                                   self.render_benchmarks,
                                   self._broker.trades)

    def close(self):
        """Utility method to clean environment before closing."""
        if self.viewer is not None:
            self.viewer.close()
