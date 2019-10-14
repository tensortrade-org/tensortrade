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
    """A trading strategy capable of self tuning, training, and evaluating with stable-baselines."""

    def __init__(self,
                 environment: TradingEnvironment,
                 model: BaseRLModel = DQN,
                 policy: Union[str, BasePolicy] = 'MlpPolicy',
                 model_kwargs: any = {},
                 **kwargs):
        """
        Arguments:
            environment: A `TradingEnvironment` instance for the agent to trade within.
            model (optional): The RL model to create the agent with. Defaults to DQN.
            policy (optional): The RL policy to train the agent's model with. Defaults to 'MlpPolicy'.
            model_kwargs (optional): Any additional keyword arguments to adjust the model.
            kwargs (optional): Optional keyword arguments to adjust the strategy.
        """
        self._model = model
        self._model_kwargs = model_kwargs

        self._environment = DummyVecEnv([lambda: environment])
        self._agent = self._model(policy, self._environment, **self._model_kwargs)

    def restore_agent(self, path: str):
        """Deserialize the strategy's learning agent from a file.

        Arguments:
            path: The `str` path of the file the agent specification is stored in.
        """
        self._agent = self._model.load(path, self._environment, self._model_kwargs)

    def save_agent(self, path: str):
        """Serialize the learning agent to a file for restoring later.

        Arguments:
            path: The `str` path of the file to store the agent specification in.
        """
        self._agent.save(path)

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, steps: int = None, episodes: int = None, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None and episodes is None:
            raise ValueError(
                'You must set the number of `steps` or `episodes` to run the strategy.')

        steps_completed = 0
        episodes_completed = 0
        average_reward = 0

        obs, state, dones = self._environment.reset(), None, [False]

        performance = {}

        while (steps is not None and steps_completed < steps) or (episodes is not None and episodes_completed < episodes):
            actions, state = self._agent.predict(obs, state=state, mask=dones)
            obs, rewards, dones, info = self._environment.step(actions)

            steps_completed += 1
            average_reward -= average_reward / steps_completed
            average_reward += rewards[0] / (steps_completed + 1)

            exchange_performance = info[0].get('exchange').performance
            performance = exchange_performance if len(exchange_performance) > 0 else performance

            if dones[0]:
                episodes_completed += 1

                if episode_callback is not None and episode_callback(self._environment._exchange.performance):
                    break

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(episodes_completed, steps_completed))
        print("Average reward: {}.".format(average_reward))

        return performance
