
import pandas as pd

from typing import Dict, Callable

from .source import DataSource, DataFrame

from tensortrade.wallets import Portfolio
from tensortrade.instruments import TradingPair


class PortfolioDataSource(DataSource):

    def __init__(self, portfolio: 'Portfolio'):
        super().__init__()
        self._portfolio = portfolio

    def next(self) -> Dict[str, any]:
        data = {}
        for w in self._portfolio.wallets:

            symbol = w.instrument.symbol

            data[symbol] = w.balance.size
            data[symbol + "_pending"] = w.locked_balance.size

        data['net_worth'] = self._portfolio.net_worth

        return data

    def has_next(self) -> bool:
        return True


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

    @property
    def prices(self):
        return self._prices

    def next(self) -> Dict[str, any]:
        data = self.data_frame_ds.next()

        prices = self.fetch(data)
        prices = {pair: price * pair for pair, price in prices.items()}
        self._prices = prices

        return data

    def has_next(self) -> bool:
        return self.data_frame_ds.has_next()

    def reset(self):
        self.data_frame_ds.reset()

        data = self.data_frame_ds.next()
        prices = self.fetch(data)
        prices = {pair: price * pair for pair, price in prices.items()}

        self._prices = prices
