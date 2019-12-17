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

import os
import gym
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines import DQN

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.strategies import TradingStrategy


class StableBaselinesTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with stable-baselines.

    Arguments:
        environments: An instance of a trading environments for the agent to trade within.
        model: The RL model to create the agent with.
            Defaults to DQN.
        policy: The RL policy to train the agent's model with.
            Defaults to 'MlpPolicy'.
        model_kwargs: Any additional keyword arguments to adjust the model.
        kwargs: Optional keyword arguments to adjust the strategy.
    """

    def __init__(self,
                 environment: TradingEnvironment,
                 model: BaseRLModel = DQN,
                 policy: Union[str, BasePolicy] = 'MlpPolicy',
                 model_kwargs: any = {},
                 **kwargs):
        self._model = model
        self._model_kwargs = model_kwargs

        self.environment = environment
        self._agent = self._model(policy, self._vectorized_environment, **self._model_kwargs)

    @property
    def environment(self) -> 'TradingEnvironment':
        """A `TradingEnvironment` instance for the agent to trade within."""
        return self._environment

    @environment.setter
    def environment(self, environment: 'TradingEnvironment'):
        self._environment = environment
        self._vectorized_environment = DummyVecEnv([lambda: environment])

    def restore_agent(self, path: str):
        """Deserialize the strategy's learning agent from a file.

        Arguments:
            path: The `str` path of the file the agent specification is stored in.
        """
        self._agent = self._model.load(path, self._vectorized_environment, self._model_kwargs)

    def save_agent(self, path: str):
        """Serialize the learning agent to a file for restoring later.

        Arguments:
            path: The `str` path of the file to store the agent specification in.
        """
        self._agent.save(path)

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def evaluate(self,
                 steps: int = None,
                 episodes=None,
                 render_mode: str = None,
                 episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None and episodes is None:
            raise ValueError(
                'You must set the number of `steps` or `episodes` to evalaute the strategy.')

        steps_completed, episodes_completed, average_reward = 0, 0, 0
        obs, state, dones = self._vectorized_environment.reset(), None, [False]
        performance = {}

        while (steps is not None and (steps == 0 or steps_completed < steps)) or (episodes is not None and episodes_completed < episodes):
            actions, state = self._agent.predict(obs, state=state, mask=dones)
            obs, rewards, dones, info = self._vectorized_environment.step(actions)

            steps_completed += 1
            average_reward -= average_reward / steps_completed
            average_reward += rewards[0] / (steps_completed + 1)

            portfolio_performance = info[0].get('portfolio').performance
            performance = portfolio_performance if len(portfolio_performance) > 0 else performance

            if render_mode is not None and self._environment.clock.step:
                self._vectorized_environment.render(mode=render_mode)

            if dones[0]:
                if episode_callback is not None and not episode_callback(performance):
                    break

                episodes_completed += 1
                obs = self._vectorized_environment.reset()

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(episodes_completed, steps_completed))
        print("Average reward: {}.".format(average_reward))

        return performance

    def _train_callback(self, _locals, _globals):
        performance = self._environment.portfolio.performance

        if self._episode_callback and self._environment.done():
            self._episode_callback(performance)

        return True

    def run(self,
            steps: int = None,
            episodes: int = None,
            render_mode: str = None,
            evaluation: bool = False,
            episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None and not evaluation:
            raise ValueError(
                "You must set the number of `steps` to train the strategy.")

        if hasattr(self._environment.exchange, '_min_time_slice') and steps < self._environment.exchange._min_time_slice:
            raise ValueError("The number of `steps` ({}) cannot be less than the environment's `_min_time_slice` ({}) ".format(
                steps, self._environment.exchange._min_time_slice))

        if evaluation:
            return self.evaluate(steps, episodes, render_mode, episode_callback)

        self._episode_callback = episode_callback

        self._agent.learn(steps, callback=self._train_callback)

        return self._environment.portfolio.performance
