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
from tensorforce.environments import Environment

from tensortrade.strategies import TradingStrategy


class TensorforceTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with Tensorforce."""

    def __init__(self, environment: 'TradingEnvironment', agent: any, max_episode_timesteps: int, agent_kwargs: any = {}, **kwargs):
        """
        Arguments:
            environment: A `TradingEnvironment` instance for the agent to trade within.
            agent: A `Tensorforce` agent or agent specification.
            save_best_agent (optional): The runner will automatically save the best agent
            kwargs (optional): Optional keyword arguments to adjust the strategy.
        """
        self._max_episode_timesteps = max_episode_timesteps
        self._save_best_agent = kwargs.get('save_best_agent', False)

        self._environment = environment
        self._tensorforce_environment = Environment.create(
            environment='gym', level=environment, max_episode_timesteps=self._max_episode_timesteps)

        self._agent = Agent.create(agent=agent,
                                   environment=self._tensorforce_environment,
                                   max_episode_timesteps=max_episode_timesteps,
                                   **agent_kwargs)

        self._runner = Runner(agent=self._agent,
                              environment=self._tensorforce_environment,
                              save_best_agent=self._save_best_agent)

    @property
    def environment(self) -> 'TradingEnvironment':
        """The `TradingEnvironment` being traded by the learning agent."""
        return self._environment

    @environment.setter
    def environment(self, environment: 'TradingEnvironment'):
        self._environment = environment
        self._tensorforce_environment = Environment.create(
            environment='gym', level=environment, max_episode_timesteps=self._max_episode_timesteps)

        self._runner = Runner(agent=self._agent,
                              environment=self._tensorforce_environment,
                              save_best_agent=self._save_best_agent)

    @property
    def agent(self) -> Agent:
        """A Tensorforce `Agent` instance that will learn the strategy."""
        return self._agent

    @agent.setter
    def agent(self, agent: any):
        self._agent = Agent.create(agent=agent, environment=self._tensorforce_environment)

        self._runner = Runner(agent=self._agent,
                              environment=self._tensorforce_environment,
                              save_best_agent=self._save_best_agent)

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

        self._runner = Runner(agent=self._agent, environment=self._tensorforce_environment)

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
        n_episodes = runner.episodes
        n_timesteps = runner.episode_timesteps
        avg_reward = np.mean(runner.episode_rewards)

        print("Finished episode {} after {} timesteps.".format(n_episodes, n_timesteps))
        print("Average episode reward: {})".format(avg_reward))

        return True

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def run(self,
            steps: int = None,
            episodes: int = None,
            render_mode: str = None,
            evaluation: bool = False,
            episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        self._runner.run(evaluation=evaluation,
                         num_timesteps=steps,
                         num_episodes=episodes,
                         callback=episode_callback)

        n_episodes = self._runner.episodes
        n_timesteps = self._runner.timesteps
        avg_reward = np.mean(self._runner.episode_rewards) \
            if self._runner.episodes > 0 else self._runner.episode_reward

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(n_episodes, n_timesteps))
        print("Average reward: {}.".format(avg_reward))

        self._runner.close()

        return self._environment.portfolio.performance
