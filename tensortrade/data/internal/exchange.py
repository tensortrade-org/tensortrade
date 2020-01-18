

from abc import abstractmethod
from typing import List

from tensortrade.instruments import TradingPair
from tensortrade.data import DataSource, Array
from tensortrade.data.stream.transform import Transform


class PriceDS(DataSource):

    def __init__(self, pair: TradingPair):
        super().__init__(str(pair))
        self._pair = pair
        self._price = None

    @property
    def price(self):
        return self._price

    @property
    def pair(self):
        return self._pair

    @abstractmethod
    def generate(self):
        pass

    def reset(self):
        self._price = None


class PriceArray(PriceDS):

    def __init__(self, pair: TradingPair, prices: List[float]):
        super().__init__(pair)
        self.prices = Array(self.name, prices)

    def generate(self):
        price = self.prices.next()
        self._price = price
        return price

    def has_next(self):
        return self.prices.has_next()

    def reset(self):
        super().reset()
        self.prices.reset()


class ExchangeDS(Transform):

    def __init__(self, name: str):
        super().__init__(name)
        self._prices = None

    @property
    def prices(self):
        return self._prices

    def transform(self, inbound_data: dict):
        pass

    def has_next(self):
        return all([price_ds.has_next() for price_ds in self.inbound])

