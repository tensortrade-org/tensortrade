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
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import OpenAIGym

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.strategies import TradingStrategy


class TensorforceTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with Tensorforce."""

    def __init__(self, environment: TradingEnvironment, agent_spec: any, **kwargs):
        """
        Arguments:
            environment: A `TradingEnvironment` instance for the agent to trade within.
            agent: A `Tensorforce` agent or agent specification.
            kwargs (optional): Optional keyword arguments to adjust the strategy.
        """
        
        exchange = environment['exchange']
        action_strategy = environment['action_strategy']
        reward_strategy = environment['reward_strategy']
        feature_pipeline = environment['feature_pipeline']
        
        self._environment = OpenAIGym(level="tensortrade-v0", 
                                      exchange=exchange,
                                      action_strategy=action_strategy,
                                      reward_strategy=reward_strategy,
                                      feature_pipeline=feature_pipeline)
        self._environment.reset()

        self._max_episode_timesteps = kwargs.get('max_episode_timesteps', None)

        self._agent = Agent.create(agent=agent_spec, environment=self._environment)

        self._runner = Runner(agent=self._agent, environment=self._environment)

    @property
    def agent(self) -> Agent:
        """A Tensorforce `Agent` instance that will learn the strategy."""
        return self._agent

    @property
    def max_episode_timesteps(self) -> int:
        """The maximum timesteps per episode."""
        return self._max_episode_timesteps

    @max_episode_timesteps.setter
    def max_episode_timesteps(self, max_episode_timesteps: int):
        self._max_episode_timesteps = max_episode_timesteps

    def restore_agent(self, directory: str, filename: str = None):
        """Deserialize the strategy's learning agent from a file.
        Arguments:
            directory: The `str` path of the directory the agent checkpoint is stored in.
            filename (optional): The `str` path of the file the agent specification is stored in.
                The `.json` file extension will be automatically appended if not provided.
        """
        self._agent = Agent.load(directory, filename=filename)

        self._runner = Runner(agent=self._agent, environment=self._environment)

    def save_agent(self, directory: str, filename: str = None, append_timestep: bool = False):
        """Serialize the learning agent to a file for restoring later.
        Arguments:
            directory: The `str` path of the directory the agent checkpoint is stored in.
            filename (optional): The `str` path of the file the agent specification is stored in.
                The `.json` file extension will be automatically appended if not provided.
            append_timestep: Whether the timestep should be appended to filename to prevent overwriting previous models.
                Defaults to `False`.
        """
        self._agent.save(directory=directory, filename=filename, append_timestep=append_timestep)

    def _finished_episode_cb(self, runner: Runner) -> bool:
        n_episodes = runner.episode
        n_timesteps = runner.episode_timestep
        avg_reward = np.mean(runner.episode_rewards)

        print("Finished episode {} after {} timesteps.".format(n_episodes, n_timesteps))
        print("Average episode reward: {})".format(avg_reward))

        return True

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, steps: int = None, episodes: int = None, evaluation: bool = True, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        self._runner.run(evaluation=evaluation,
                         num_timesteps=steps,
                         num_episodes=episodes,
                         max_episode_timesteps=self._max_episode_timesteps,
                         callback=episode_callback)

        n_episodes = self._runner.episodes
        n_timesteps = self._runner.timesteps
        avg_reward = np.mean(self._runner.episode_rewards)

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(n_episodes, n_timesteps))
        print("Average reward: {}.".format(avg_reward))

        self._runner.close()

        return self._environment.environment._exchange._performance
