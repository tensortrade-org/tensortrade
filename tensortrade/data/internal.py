
import pandas as pd

from typing import Dict, Callable

from .source import DataSource, DataFrame

from tensortrade.wallets import Portfolio
from tensortrade.instruments import TradingPair


class PortfolioDataSource(DataSource):

    def __init__(self,
                 portfolio: 'Portfolio',
                 fetch: Callable[[dict], Dict[TradingPair, float]]):
        super().__init__()
        self._portfolio = portfolio
        self._fetch = fetch

    def call(self, data: dict) -> Dict[str, any]:
        portfolio_data = {}

        net_worth = 0
        for w in self._portfolio.wallets:

            symbol = w.instrument.symbol

            portfolio_data[symbol] = w.balance.size
            portfolio_data[symbol + "_pending"] = w.locked_balance.size
            total_balance = w.balance.size + w.locked_balance.size

            if w.instrument == self._portfolio.base_instrument:
                net_worth += total_balance
            else:
                pair = self._portfolio.base_instrument / w.instrument
                price = self._fetch(data)[pair]

                net_worth += price * total_balance

        portfolio_data['net_worth'] = net_worth

        return portfolio_data

    def has_next(self) -> bool:
        return True

    def reset(self) -> bool:
        pass


class ExchangeDataSource(DataSource):

    def __init__(self,
                 data_frame: pd.DataFrame,
                 fetch: Callable[[dict], Dict[TradingPair, float]]):
        super().__init__()
        self.data_frame_ds = DataFrame(data_frame)
        self.fetch = fetch

        data = self.data_frame_ds.next()
        prices = self.fetch(data)
        prices = {pair: price * pair for pair, price in prices.items()}
        self._prices = prices

        self.data_frame_ds.reset()

    @property
    def prices(self):
        return self._prices

    def call(self, data) -> Dict[str, any]:
        data = self.data_frame_ds.next()
        prices = self.fetch(data)
        prices = {pair: price * pair for pair, price in prices.items()}
        self._prices = prices
        return data

    def has_next(self) -> bool:
        return self.data_frame_ds.has_next()

    def reset(self):
        data = self.data_frame_ds.next()
        prices = self.fetch(data)
        prices = {pair: price * pair for pair, price in prices.items()}
        self._prices = prices

        self.data_frame_ds.reset()
