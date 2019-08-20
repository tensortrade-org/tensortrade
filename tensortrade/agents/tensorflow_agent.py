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
# limitations under the License

import pandas as pd
import tensorflow as tf

from ray import tune
from ray.tune import grid_search
from typing import Union, Callable, List

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.models.rl import TensorflowModel
from tensortrade.agents import TradingAgent

""" [WIP] """


class TensorflowAgent(TradingAgent):
    """A trading agent capable of self tuning, training, and evaluating with the RLlib API and Tensorflow 2."""

    def __init__(self, env: TradingEnvironment, feature_pipeline: FeaturePipeline, agent: model: TensorflowModel, **kwargs):
        """
        Arguments:
            env: A `TradingEnvironment` instance for the agent to trade within.
            feature_pipeline: A `FeaturePipeline` instance of feature transformations.
            model: A `TensorflowModel` instance to be used by the agent.
        """
        super().__init__(env=env, feature_pipeline=feature_pipeline)

        self._model = model
        self._vf_share_layers = kwargs.get('vf_share_layers', True)
        self._learning_rate = kwargs.get('learning_rate', grid_search([1e-2, 1e-4, 1e-6]))
        self._num_workers = kwargs.get('num_workers', 1)

    def tune(self, steps_per_train: int, steps_per_test: int, step_cb: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        pass

    def train(self, steps: int, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        tune.run(
            self._agent,
            stop={
                "timesteps_total": steps,
            },
            config={
                "env": self._env,
                "model": {
                    "custom_model": self._model,
                },
                "vf_share_layers": self._vf_share_layers,
                "lr": self._learning_rate,
                "num_workers": self._num_workers,
            },
        )

    def evaluate(self, steps: int, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        pass

    def get_action(self, observation: pd.DataFrame) -> Union[float, List[float]]:
        pass
