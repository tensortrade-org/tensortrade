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

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.strategies import TradingStrategy


class TensorforceTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with Tensorforce."""

    def __init__(self, env: TradingEnvironment, agent: Agent):
        """
        Arguments:
            env: A `TradingEnvironment` instance for the agent to trade within.
            agent: A Tensorforce `Agent` instance that will learn the strategy.
        """
        self._env = env
        self._agent = agent

    @property
    def agent(self) -> Agent:
        """A Tensorforce `Agent` instance that will learn the strategy."""
        return self._agent

    @agent.setter
    def agent(self, agent: Agent):
        self._agent = agent

    @staticmethod
    def restore(self, path: str):
        raise NotImplementedError

    def tune(self, steps_per_train: int, steps_per_test: int, step_cb: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def train(self, steps: int, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, steps: int, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, path: str):
        raise NotImplementedError
