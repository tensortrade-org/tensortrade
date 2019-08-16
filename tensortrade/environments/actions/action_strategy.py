from typing import Union
from abc import abstractmethod
from gym import spaces

from tensortrade.exchanges import AssetExchange


class ActionStrategy(object):
    def __init__(self, action_space_shape: Union[int, tuple],
                 continuous_action_space: bool = False):
        self.action_space_shape = action_space_shape
        self.continuous_action_space = continuous_action_space

    def set_dtype(self, dtype):
        self.dtype = dtype

    def action_space(self):
        if self.continuous_action_space:
            if type(self.action_space_shape) is not tuple:
                raise ValueError(
                    '`action_space_shape` must be a `tuple` when `continuous_action_space` is `True`.')

            return spaces.Box(low=0, high=1, shape=self.action_space_shape, dtype=self.dtype)
        else:
            if type(self.action_space_shape) is not int:
                raise ValueError(
                    '`action_space_shape` must be an `int` when `continuous_action_space` is `False`.')

            return spaces.Discrete(self.action_space_shape)

    def reset(self):
        pass

    @abstractmethod
    def get_trade(self, action: Union[int, tuple], exchange: AssetExchange):
        raise NotImplementedError
