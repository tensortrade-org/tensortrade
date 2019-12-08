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

import tensortrade.exchanges as exchanges
import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.features as features

from gym import spaces
from typing import Union, Tuple, List, Dict

from tensortrade.actions import ActionScheme
from tensortrade.rewards import RewardScheme
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline
from tensortrade.orders import Broker, Order

if importlib.util.find_spec("matplotlib") is not None:
    from tensortrade.environments.render import MatplotlibTradingChart


class TradingEnvironment(gym.Env):
    """A trading environments made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self,
                 exchange: Union[Exchange, str],
                 action_scheme: Union[ActionScheme, str],
                 reward_scheme: Union[RewardScheme, str],
                 feature_pipeline: Union[FeaturePipeline, str] = None,
                 **kwargs):
        """
        Arguments:
            exchange: The `Exchange` that will be used to feed data from and execute trades within.
            action_scheme:  The component for transforming an action into an `Order` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            feature_pipeline (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """
        super().__init__()

        self._exchange = exchanges.get(exchange) if isinstance(exchange, str) else exchange
        self._action_scheme = actions.get(action_scheme) if isinstance(
            action_scheme, str) else action_scheme
        self._reward_scheme = rewards.get(reward_scheme) if isinstance(
            reward_scheme, str) else reward_scheme
        self._feature_pipeline = features.get(feature_pipeline) if isinstance(
            feature_pipeline, str) else feature_pipeline

        if feature_pipeline is not None:
            self._exchange.feature_pipeline = feature_pipeline

        self._action_scheme.exchange = self._exchange
        self._reward_scheme.exchange = self._exchange

        self.observation_space = self._exchange.observation_space
        self.action_space = self._action_scheme.action_space

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.viewer = None

        self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
        self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))
        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        self.reset()

    @property
    def exchange(self) -> Exchange:
        """The `Exchange` that will be used to feed data from and execute trades within."""
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: Exchange):
        self._exchange = exchange

    @property
    def episode_trades(self) -> pd.DataFrame:
        """A `pandas.DataFrame` of trades made this episode."""
        return self.exchange.trades

    @property
    def action_scheme(self) -> ActionScheme:
        """The component for transforming an action into an `Order` at each time step."""
        return self._action_scheme

    @action_scheme.setter
    def action_scheme(self, action_scheme: ActionScheme):
        self._action_scheme = action_scheme

    @property
    def reward_scheme(self) -> RewardScheme:
        """The component for determining the reward at each time step."""
        return self._reward_scheme

    @reward_scheme.setter
    def reward_scheme(self, reward_scheme: RewardScheme):
        self._reward_scheme = reward_scheme

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """The feature pipeline to pass the observations through."""
        return self._exchange.feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self._exchange.feature_pipeline = feature_pipeline

    def _take_action(self, action: int) -> Order:
        """Determines a specific trade to be taken and executes it within the exchange.

        Arguments:
            action: The int provided by the agent to map to a trade action for this timestep.

        Returns:
            The order created by the agent this timestep, if any.
        """
        order = self._action_scheme.get_order(action, self._exchange)

        if order:
            self._exchange.submit_to_broker(order)

        return order

    def _next_observation(self) -> np.ndarray:
        """Returns the next observation from the exchange.

        Returns:
            The observation provided by the environments's exchange, often OHLCV or tick trade history data points.
        """
        observation = self._exchange.next_observation()

        if len(observation) != 0:
            observation = observation[0]

        observation = np.nan_to_num(observation)

        return observation

    def _get_reward(self) -> float:
        """Returns the reward for the current timestep.

        Returns:
            A float corresponding to the benefit earned by the action taken this step.
        """
        reward = self._reward_scheme.get_reward(current_step=self._current_step)

        print('Reward: ', reward)

        reward = np.nan_to_num(reward)

        if np.bitwise_not(np.isfinite(reward)):
            raise ValueError('Reward returned by the reward scheme must by a finite float.')

        return reward

    def _done(self) -> bool:
        """Returns whether or not the environments is done and should be restarted.

        Returns:
            A boolean signaling whether the environments is done and should be restarted.
        """
        lost_90_percent_net_worth = self._exchange.portfolio.profit_loss < 0.1
        return lost_90_percent_net_worth or not self._exchange.has_next_observation

    def _info(self, order: Order) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.

        Args:
            order: The order created during the currente timestep.

        Returns:
            info: A dictionary containing the exchange used, the current timestep, and the filled trade, if any.
        """
        return {
            'current_step': self._current_step,
            'exchange': self._exchange,
            'order': order,
        }

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environments based on the specified action.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environments's exchange, often OHLCV or tick trade history data points.
            reward (float): An amount corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environments is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """
        order = self._take_action(action)

        observation = self._next_observation()
        reward = self._get_reward()
        done = self._done()
        info = self._info(order)

        self._current_step += 1

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """Resets the state of the environments and returns an initial observation.

        Returns:
            The episode's initial observation.
        """
        self._action_scheme.reset()
        self._reward_scheme.reset()
        self._exchange.reset()

        self._current_step = 0

        observation = self._next_observation()

        self._current_step = 1

        return observation

    def render(self, mode='none'):
        """Renders the environment via matplotlib."""
        if mode == 'log':
            self.logger.info('Price: ' + str(self.exchange._current_price()))
            self.logger.info('Net worth: ' + str(self.exchange.performance[-1]['net_worth']))
        elif mode == 'chart':
            if self.viewer is None and hasattr(self.exchange, '_pre_transformed_data'):
                self.viewer = MatplotlibTradingChart(self.exchange._pre_transformed_data)

            if self.viewer is not None:
                self.viewer.render(self._current_step - 1,
                                   self.exchange.performance['net_worth'].values,
                                   self.render_benchmarks,
                                   self.exchange.trades)

    def close(self):
        """Utility method to clean environment before closing."""
        if self.viewer is not None:
            self.viewer.close()
