import numpy as np

from abc import ABCMeta, abstractmethod


class AssetExchange(object, metaclass=ABCMeta):
    dtype: type = np.float16

    @abstractmethod
    def __init__(self, commission_percent: float, base_precision: float, asset_precision: float):
        self.commission_percent = commission_percent
        self.base_precision = base_precision
        self.asset_precision = asset_precision

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_max_allowed_slippage_percent(self, slippage_percent):
        self.max_allowed_slippage_percent = slippage_percent

    def reset(self):
        pass

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def current_price(self):
        raise NotImplementedError

    @abstractmethod
    def execute_trade(self):
        raise NotImplementedError

    @abstractmethod
    def has_next_observation(self):
        raise NotImplementedError

    @abstractmethod
    def next_observation(self):
        raise NotImplementedError
