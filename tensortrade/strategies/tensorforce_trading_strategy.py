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

from tensortrade.strategies import TradingStrategy


class TensorforceTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with Tensorforce."""

    def __init__(self, environment: 'TradingEnvironment', agent_spec: Dict, network_spec: Dict, **kwargs):
        """
        Arguments:
            environment: A `TradingEnvironment` instance for the agent to trade within.
            agent_spec: A specification dictionary for the `Tensorforce` agent.
            network_spec: A specification dictionary for the `Tensorforce` agent's model network.
            kwargs (optional): Optional keyword arguments to adjust the strategy.
        """
        self._environment = environment

        self._max_episode_timesteps = kwargs.get('max_episode_timesteps', None)

        self._agent_spec = agent_spec
        self._network_spec = network_spec

        self._agent = Agent.from_spec(spec=agent_spec,
                                      kwargs=dict(network=network_spec,
                                                  states=environment.states,
                                                  actions=environment.actions))

        self._runner = Runner(agent=self._agent, environment=environment)

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

    def restore_agent(self, path: str, model_path: str = None):
        """Deserialize the strategy's learning agent from a file.

        Arguments:
            path: The `str` path of the file the agent specification is stored in.
                The `.json` file extension will be automatically appended if not provided.
            model_path (optional): The `str` path of the file or directory the agent checkpoint is stored in.
                If not provided, the `model_path` will default to `{path_without_dot_json}/agents`.
        """
        path_with_ext = path if path.endswith('.json') else '{}.json'.format(path)

        with open(path_with_ext) as json_file:
            spec = json.load(json_file)

            self._agent_spec = spec.agent
            self._network_spec = spec.network

        self._agent = Agent.from_spec(spec=self._agent_spec,
                                      kwargs=dict(network=self._network_spec,
                                                  states=self._environment.states,
                                                  actions=self._environment.actions))

        path_without_ext = path_with_ext.replace('.json', '')
        model_path = model_path or '{}/agent'.format(path_without_ext)

        self._agent.restore_model(file=model_path)

        self._runner = Runner(agent=self._agent, environment=self._environment)

    def save_agent(self, path: str, model_path: str = None, append_timestep: bool = False):
        """Serialize the learning agent to a file for restoring later.

        Arguments:
            path: The `str` path of the file to store the agent specification in.
                The `.json` file extension will be automatically appended if not provided.
            model_path (optional): The `str` path of the directory to store the agent checkpoints in.
                If not provided, the `model_path` will default to `{path_without_dot_json}/agents`.
            append_timestep: Whether the timestep should be appended to filename to prevent overwriting previous models.
                Defaults to `False`.
        """
        path_with_ext = path if path.endswith('.json') else '{}.json'.format(path)

        spec = {'agent': self._agent_spec, 'network': self._network_spec}

        with open(path_with_ext, 'w') as json_file:
            json.dump(spec, json_file)

        path_without_ext = path_with_ext.replace('.json', '')
        model_path = model_path or '{}/agent'.format(path_without_ext)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self._agent.save_model(directory=model_path, append_timestep=True)

    def _finished_episode_cb(self, runner: Runner) -> bool:
        n_episodes = runner.episode
        n_timesteps = runner.episode_timestep
        avg_reward = np.mean(runner.episode_rewards)

        print("Finished episode {} after {} timesteps.".format(n_episodes, n_timesteps))
        print("Average episode reward: {})".format(avg_reward))

        return True

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, steps: int = None, episodes: int = None, testing: bool = True, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        self._runner.run(testing=testing,
                         num_timesteps=steps,
                         num_episodes=episodes,
                         max_episode_timesteps=self._max_episode_timesteps,
                         episode_finished=episode_callback)

        n_episodes = self._runner.episode
        n_timesteps = self._runner.timestep
        avg_reward = np.mean(self._runner.episode_rewards)

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(n_episodes, n_timesteps))
        print("Average reward: {}.".format(avg_reward))

        self._runner.close()

        return self._environment._exchange.performance
