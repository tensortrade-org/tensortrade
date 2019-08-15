import gin
import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline


@gin.configurable
class TradingAgent(object, metaclass=ABCMeta):
    """An abstract base class for trading agents capable of self tuning, training, and evaluating."""

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
        """A `FeaturePipeline` instance of feature transformations."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self.feature_pipeline = feature_pipeline

    @gin.configurable
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

    @gin.configurable
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

    @gin.configurable
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
    def get_action(self, observation: pd.DataFrame) -> Union[float, List[float]]:
        """Determine an action based on a specific observation.

        Args:
            observation: A `pandas.DataFrame` corresponding to an observation within the environment.

        Returns:
            An action whose type depends on the action space of the environment.
        """
        raise NotImplementedError
