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
import pandas as pd
import numpy as np

from gym import spaces
from typing import Union, Tuple, List

from tensortrade.actions import ActionStrategy, TradeActionUnion
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import RewardStrategy
from tensortrade.exchanges import InstrumentExchange
from tensortrade.trades import Trade

TensorForceStateType = Union[bool, int, float]
TensorForceStateShape = Union[int, List[int], Tuple[int, ...]]
TensorForceMinMaxValue = Union[int, float]


class TradingEnvironment(gym.Env):
    """A trading environment made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self,
                 exchange: InstrumentExchange,
                 action_strategy: ActionStrategy,
                 reward_strategy: RewardStrategy,
                 feature_pipeline: FeaturePipeline = None,
                 **kwargs):
        """
        Arguments:
            exchange: The `InstrumentExchange` that will be used to feed data from and execute trades within.
            feature_pipeline: A `FeaturePipeline` instance of feature transformations.
            action_strategy:  The strategy for transforming an action into a `Trade` at each timestep.
            reward_strategy: The strategy for determining the reward at each timestep.
            kwargs (optional): Additional arguments for tuning the environment, logging, etc.
        """

        super().__init__()

        self._exchange = exchange
        self._action_strategy = action_strategy
        self._reward_strategy = reward_strategy
        self._feature_pipeline = feature_pipeline

        self._action_strategy.exchange = self._exchange
        self._reward_strategy.exchange = self._exchange

        self.observation_space = self._exchange.observation_space
        self.action_space = self._action_strategy.action_space

        self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
        self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

    @property
    def exchange(self) -> InstrumentExchange:
        """The `InstrumentExchange` that will be used to feed data from and execute trades within."""
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: InstrumentExchange):
        self._exchange = exchange

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """A pipeline of feature transformations to be applied to observations from the environment."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self._feature_pipeline = feature_pipeline

    @property
    def action_strategy(self) -> ActionStrategy:
        """The strategy for transforming an action into a `Trade` at each timestep."""
        return self._action_strategy

    @action_strategy.setter
    def action_strategy(self, action_strategy: ActionStrategy):
        self._action_strategy = action_strategy

    @property
    def reward_strategy(self) -> RewardStrategy:
        """The strategy for determining the reward at each timestep"""
        return self._reward_strategy

    @reward_strategy.setter
    def reward_strategy(self, reward_strategy: RewardStrategy):
        self._reward_strategy = reward_strategy

    @property
    def states(self) -> Tuple[TensorForceStateType, TensorForceStateShape]:
        """The state space specification, required for `tensorforce` agents.

        The tuple contains the following attributes:
            - type: Either 'bool', 'int', or 'float'.
            - shape: The shape of the space. An `int` or `list`/`tuple` of `int`s.
        """
        from tensorforce.contrib.openai_gym import OpenAIGym
        return OpenAIGym.state_from_space(self.observation_space)

    @property
    def actions(self) -> Tuple[TensorForceStateType, TensorForceStateShape, int, TensorForceMinMaxValue, TensorForceMinMaxValue]:
        """The action space specification, required for `tensorforce` agents.

        The tuple contains the following attributes:
            - type: Either 'bool', 'int', or 'float'.
            - shape: The shape of the space. An `int` or `list`/`tuple` of `int`s.
            - num_actions (required if type == 'int'): The number of discrete actions.
            - min_value (optional if type == 'float'): An `int` or `float`. Defaults to `None`.
            - max_value (optional if type == 'float'): An `int` or `float`. Defaults to `None`.
        """
        from tensorforce.contrib.openai_gym import OpenAIGym
        return OpenAIGym.action_from_space(self.action_space)

    def _take_action(self, action: TradeActionUnion) -> Trade:
        """Determines a specific trade to be taken and executes it within the exchange.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            A tuple containing the (fill_amount, fill_price) of the executed trade.
        """
        executed_trade = self._action_strategy.get_trade(action=action)

        filled_trade = self._exchange.execute_trade(executed_trade)

        return executed_trade, filled_trade

    def _next_observation(self, trade: Trade) -> pd.DataFrame:
        """Returns the next observation from the exchange.

        Returns:
            The observation provided by the environment's exchange, often OHLCV or tick trade history data points.
        """
        self._current_step += 1

        observation = self._exchange.next_observation()
        observation = observation.fillna(0, axis=0)

        if self._feature_pipeline is not None:
            observation = self._feature_pipeline.transform(observation)

        return observation

    def _get_reward(self, trade: Trade) -> float:
        """Returns the reward for the current timestep.

        Returns:
            A float corresponding to the benefit earned by the action taken this step.
        """
        reward: float = self._reward_strategy.get_reward(
            current_step=self._current_step, trade=trade)

        if not np.isfinite(reward):
            raise ValueError('Reward returned by the reward strategy must by a finite float.')

        return reward

    def _done(self) -> bool:
        """Returns whether or not the environment is done and should be restarted.

        Returns:
            A boolean signaling whether the environment is done and should be restarted.
        """
        lost_90_percent_net_worth = self._exchange.profit_loss_percent < 0.1

        return lost_90_percent_net_worth or not self._exchange.has_next_observation

    def _info(self, executed_trade: Trade, filled_trade: Trade) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.

        Returns:
            info: A dictionary containing the exchange used, the current timestep, and the filled trade, if any.
        """
        return {'current_step': self._current_step,
                'exchange': self._exchange,
                'executed_trade': executed_trade,
                'filled_trade': filled_trade}

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environment based on the specified action.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environment's exchange, often OHLCV or tick trade history data points.\n
            reward (float): An amount corresponding to the benefit earned by the action taken this timestep.\n
            done (bool): If `True`, the environment is complete and should be restarted.\n
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """

        executed_trade, filled_trade = self._take_action(action)

        observation = self._next_observation(filled_trade)
        reward = self._get_reward(filled_trade)
        done = self._done()
        info = self._info(executed_trade, filled_trade)

        return observation, reward, done, info

    def execute(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environment based on the specified action, required for `tensorforce` agents.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (np.ndarray): Provided by the environment's exchange, often OHLCV or tick trade history data points.\n
            done (bool): If `True`, the environment is complete and should be restarted.\n
            reward (float): An amount corresponding to the benefit earned by the action taken this timestep.\n
        """
        observation, done, reward, _ = self.step(action)
        return observation.values, reward, done

    def reset(self) -> pd.DataFrame:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation: the initial observation.
        """

        self._action_strategy.reset()
        self._reward_strategy.reset()
        self._exchange.reset()

        self._current_step = 0

        return self._next_observation(Trade('N/A', 'hold', 0, 0))

    def render(self, mode='none'):
        """Renders the environment."""
        pass
