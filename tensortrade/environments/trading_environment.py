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
from typing import Union, Tuple

from tensortrade.actions import ActionStrategy, TradeActionUnion
from tensortrade.rewards import RewardStrategy
from tensortrade.exchanges import AssetExchange
from tensortrade.trades import Trade


class TradingEnvironment(gym.Env):
    """A trading environment made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self,
                 exchange: AssetExchange,
                 action_strategy: ActionStrategy,
                 reward_strategy: RewardStrategy,
                 **kwargs):
        """
        Args:
            exchange: The `AssetExchange` that will be used to feed data from and execute trades within.
            action_strategy:  The strategy for transforming an action into a `Trade` at each timestep.
            reward_strategy: The strategy for determining the reward at each timestep.
            kwargs (optional): Additional arguments for tuning the environment, logging, etc.
        """

        super().__init__()

        self.exchange = exchange
        self.action_strategy = action_strategy
        self.reward_strategy = reward_strategy

        self.action_strategy.set_exchange(self.exchange)
        self.reward_strategy.set_exchange(self.exchange)

        self.observation_space = self.exchange.observation_space
        self.action_space = self.action_strategy.action_space

        self.logger_name: int = kwargs.get('logger_name', __name__)
        self.log_level: int = kwargs.get('log_level', logging.DEBUG)
        self.disable_tensorflow_logs: bool = kwargs.get('disable_tensorflow_logger', True)

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)

        logging.getLogger('tensorflow').disabled = self.disable_tensorflow_logs

    def _take_action(self, action: TradeActionUnion) -> Trade:
        """Determines a specific trade to be taken and executes it within the exchange.

        Args:
            action: The trade action provided by the agent for this timestep.

        Returns:
            A tuple containing the (fill_amount, fill_price) of the executed trade.
        """
        trade = self.action_strategy.get_trade(action=action, exchange=self.exchange)

        filled_trade = self.exchange.execute_trade(trade)

        return filled_trade

    def _next_observation(self, trade: Trade) -> pd.DataFrame:
        """Returns the next observation from the exchange.

        Returns:
            The observation provided by the environment's exchange, often OHLCV or tick trade history data points.
        """
        self.current_step += 1

        observation = self.exchange.next_observation()
        observation = observation.fillna(0, axis=1)

        observation['fill_amount'] = [trade.amount]
        observation['fill_price'] = [trade.price]

        return observation

    def _get_reward(self, trade: Trade) -> float:
        """Returns the reward for the current timestep.

        Returns:
            A float corresponding to the benefit earned by the action taken this step.
        """
        reward: float = self.reward_strategy.get_reward(current_step=self.current_step, trade=trade)

        return reward if np.isfinite(reward) else 0

    def _done(self) -> bool:
        """Returns whether or not the environment is done and should be restarted.

        Returns:
            A boolean signaling whether the environment is done and should be restarted.
        """
        lost_90_percent_net_worth = self.exchange.profit_loss_percent < 0.1
        has_next_obs: bool = self.exchange.has_next_observation()

        return lost_90_percent_net_worth or not has_next_obs

    def _info(self) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.

        Returns:
            info: A dictionary containing the exchange used and the current timestep.
        """
        return {'current_step': self.current_step, 'exchange': self.exchange}

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environment based on the specified action.

        Args:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environment's exchange, often OHLCV or tick trade history data points.
            reward (float): An amount corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environment is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """

        filled_trade = self._take_action(action)

        observation = self._next_observation(filled_trade)
        reward = self._get_reward(filled_trade)
        done = self._done()
        info = self._info()

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation: the initial observation.
        """

        self.action_strategy.reset()
        self.reward_strategy.reset()
        self.exchange.reset()

        self.current_step = 0

        return self._next_observation(Trade('N/A', 'hold', 0, 0))

    def render(self, mode='none'):
        """Renders the environment."""
        pass


gym.register(
    id='TradingEnvironment-v0',
    entry_point='tensortrade.environments.trading_environment:TradingEnvironment',
)
