import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import Dict


class AssetExchange(object, metaclass=ABCMeta):
    """Abstract base class for asset exchanges"""
    @abstractmethod
    def __init__(self):
        pass

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_max_allowed_slippage_percent(self, max_allowed_slippage_percent):
        self.max_allowed_slippage_percent = max_allowed_slippage_percent

    def net_worth(self, output_symbol) -> float:
        """Calculate the net worth of the current account in this exchange

            # Arguments
            output_symbol: the notional value, that should be used to display the account value

            # Returns
            the total portfolio value of this account
        """
        net_worth = self.balance(symbol=output_symbol)

        portfolio = self.portfolio()

        if not portfolio:
            return net_worth

        for symbol, amount in portfolio.items():
            current_price = self.current_price(
                symbol=symbol, output_symbol=output_symbol)
            net_worth += current_price * amount

        return net_worth

    def profit_loss_percent(self, output_symbol) -> float:
        """Calculate the percentage change since the initial balance in the output_symbol notional value"""
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
