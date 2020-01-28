import numpy as np

from abc import ABCMeta, abstractmethod

from tensortrade.base import Identifiable


class Agent(Identifiable, metaclass=ABCMeta):

    @abstractmethod
    def restore(self, path: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        raise NotImplementedError()

    @abstractmethod
    def train(self,
              n_steps: int = None,
              n_episodes: int = 10000,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs):
        raise NotImplementedError()
