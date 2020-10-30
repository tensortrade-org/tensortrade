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


import numpy as np

from abc import ABCMeta, abstractmethod

from tensortrade.core import Identifiable


class Agent(Identifiable, metaclass=ABCMeta):

    @abstractmethod
    def restore(self, path: str, **kwargs):
        """Restore the agent from the file specified in `path`."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str, **kwargs):
        """Save the agent to the directory specified in `path`."""
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """Get an action for a specific state in the environment."""
        raise NotImplementedError()

    @abstractmethod
    def train(self,
              n_steps: int = None,
              n_episodes: int = 10000,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        """Train the agent in the environment and return the mean reward."""
        raise NotImplementedError()
