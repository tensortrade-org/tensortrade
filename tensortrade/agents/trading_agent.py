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

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.actions import TradeActionUnion


class TradingAgent(object, metaclass=ABCMeta):
    """An abstract trading agent capable of self tuning, training, and evaluating."""

    @abstractmethod
    def __init__(self, env: TradingEnvironment, feature_pipeline: FeaturePipeline):
        """
        Args:
            env: A `TradingEnvironment` instance for the agent to trade within.
            feature_pipeline: A `FeaturePipeline` instance of feature transformations.
        """
        self._env = env
        self._feature_pipeline = feature_pipeline

    @property
    def env(self):
        """A `TradingEnvironment` instance for the agent to trade within."""
        return self._env

    @env.setter
    def env(self, env: TradingEnvironment):
        self._env = env

    @property
    def feature_pipeline(self):
        """A pipeline of feature transformations to be applied to observations from the environment."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self.feature_pipeline = feature_pipeline

    @abstractmethod
    def tune(self, steps_per_train: int, steps_per_test: int, step_cb: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        """Tune the agent's hyper-parameters and feature set for the environment.

        Args:
            steps_per_train: The number of steps per training of each hyper-parameter set.
            steps_per_test: The number of steps per evaluation of each hyper-parameter set.
            step_cb (optional): A callback function for monitoring progress of the tuning process.
                step_cb(pd.DataFrame) -> bool: A history of the agent's trading performance is passed on each iteration.
                If the callback returns `True`, the training process will stop early.

        Returns:
            A history of the agent's trading performance during tuning
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, steps: int, callback: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        """Train the agent's underlying model on the environment.

        Args:
            steps: The number of steps to train the model within the environment.
            step_cb (optional): A callback function for monitoring progress of the training process.
                step_cb(pd.DataFrame) -> bool: A history of the agent's trading performance is passed on each iteration.
                If the callback returns `True`, the training process will stop early.

        Returns:
            A history of the agent's trading performance during training
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, steps: int, callback: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        """Evaluate the agent's performance within the environment.

        Args:
            steps: The number of steps to train the model within the environment.
            step_cb (optional): A callback function for monitoring progress of the evaluation process.
                step_cb(pd.DataFrame) -> bool: A history of the agent's trading performance is passed on each iteration.
                If the callback returns `True`, the training process will stop early.

        Returns:
            A history of the agent's trading performance during evaluation
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self, observation: pd.DataFrame) -> TradeActionUnion:
        """Determine an action based on a specific observation.

        Args:
            observation: A `pandas.DataFrame` corresponding to an observation within the environment.

        Returns:
            An action whose type depends on the action space of the environment.
        """
        raise NotImplementedError
