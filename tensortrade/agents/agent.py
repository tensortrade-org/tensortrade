import numpy as np

from abc import ABCMeta, abstractmethod

from tensortrade.base import Identifiable


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
