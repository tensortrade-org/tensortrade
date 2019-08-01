import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import Dict


class AssetExchange(object, metaclass=ABCMeta):
    dtype: type = np.float16

    @abstractmethod
    def __init__(self, commission_percent: float, base_precision: float, asset_precision: float):
        self.commission_percent = commission_percent
        self.base_precision = base_precision
        self.asset_precision = asset_precision

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_max_allowed_slippage_percent(self, max_allowed_slippage_percent):
        self.max_allowed_slippage_percent = max_allowed_slippage_percent

    def net_worth(self, output_symbol) -> float:
        net_worth = self.balance(symbol=output_symbol)

        for symbol, amount in dir(self.portfolio()).items():
            current_price = self.current_price(symbol=symbol, output_symbol=output_symbol)
            net_worth += current_price * amount

        return net_worth

    def profit_loss_percent(self, output_symbol) -> float:
        return float(self.net_worth(output_symbol=output_symbol) / self.initial_balance(symbol=output_symbol))

    @abstractmethod
    def initial_balance(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def balance(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def portfolio(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def trades(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def performance(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def observation_space(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def current_price(self, symbol: str, output_symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def has_next_observation(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_observation(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def execute_trade(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
