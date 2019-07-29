import numpy as np

from typing import Dict
from abc import ABCMeta, abstractmethod
from gym import spaces

from trader.exchanges import AssetExchange


class ActionStrategy(object, metaclass=ABCMeta):
    dtype: type = np.float16

    def __init__(action_space_shape: tuple | int,
                 continuous_action_space: bool = False):
        self.action_space_shape = action_space_shape
        self.continuous_action_space = continuous_action_space

    def set_dtype(self, dtype):
        self.dtype = dtype

    def action_space(self):
        if self.continuous_action_space:
            if type(self.action_space_shape) is not tuple:
                raise ValueError('`action_space_shape` must be a `tuple` when `continuous_action_space` is `True`.')

            return spaces.Box(low=0, high=1, shape=self.action_space_shape, dtype=self.dtype)
        else:
            if type(self.action_space_shape) is not int:
                raise ValueError('`action_space_shape` must be an `int` when `continuous_action_space` is `False`.')

            return spaces.Discrete(self.action_space_shape)

    def reset():
        pass

    @abstractmethod
    def suggest_trade(self, action: int | tuple, balance: float, assets_held: Dict[str, float], exchange: AssetExchange):
        raise NotImplementedError
