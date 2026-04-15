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


from deprecated import deprecated

import numpy as np

from abc import ABCMeta, abstractmethod

from tensortrade.core import Identifiable


@deprecated(version='1.0.4', reason="Builtin agents are being deprecated in favor of external implementations (ie: Ray)")
class Agent(Identifiable, metaclass=ABCMeta):
    """An abstract base class for defining reinforcement learning agents.
    
    Agents are responsible for interacting with the environment, querying policies
    to select actions based on observed states, and training their internal models.
    """

    @abstractmethod
    def restore(self, path: str, **kwargs):
        """Restore the agent from a saved model at the specified file path.
        
        Args:
            path (str): The file path where the agent's model is stored.
            **kwargs: Additional keyword arguments for the restoration process.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str, **kwargs):
        """Save the current agent's model to the specified file path.
        
        Args:
            path (str): The file path where the agent's model will be saved.
            **kwargs: Additional keyword arguments for the saving process.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """Get an action for a specific state in the environment.
        
        Args:
            state (np.ndarray): The current state observation from the environment.
            **kwargs: Additional keyword arguments for action selection.
            
        Returns:
            int: The chosen action for the given state.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self,
              n_steps: int = None,
              n_episodes: int = 10000,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        """Train the agent in the environment and return the mean reward.
        
        Args:
            n_steps (int, optional): The maximum number of steps to train the agent. Defaults to None.
            n_episodes (int, optional): The number of episodes to train the agent. Defaults to 10000.
            save_every (int, optional): The frequency to save the agent's model. Defaults to None.
            save_path (str, optional): The directory path to save the agent during training. Defaults to None.
            callback (callable, optional): A callback function to be called periodically during training. Defaults to None.
            **kwargs: Additional keyword arguments for the training process.
            
        Returns:
            float: The mean reward obtained by the agent over the training period.
        """
        raise NotImplementedError()
