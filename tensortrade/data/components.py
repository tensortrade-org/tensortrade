

from typing import Dict

from .source import DataSource

from tensortrade.wallets import Portfolio


class PortfolioDataSource(DataSource):

    def __init__(self, portfolio: 'Portfolio'):
        super().__init__('portfolio')
        self._portfolio = portfolio

    def next(self) -> Dict[str, any]:
        data = {}
        for w in self._portfolio.wallets:

            symbol = w.instrument.symbol

            data[symbol] = w.balance.size
            data[symbol + "_pending"] = w.locked_balance.size

        return data

    def has_next(self) -> bool:
        return True



