
from typing import List, Union

from .order_criteria import OrderCriteria


class LimitOrderCriteria(OrderCriteria):
    """An order criteria that allows execution when the quote price for a trading pair is at or below a specific price."""

    def __init__(self, pairs: Union[List['TradingPair'], 'TradingPair'], limit_price: float):
        super().__init__(pairs=pairs)

        self.limit_price = limit_price

    def is_executable(self, pair: 'TradingPair', exchange: 'Exchange') -> bool:
        return self.is_pair_enabled(pair) and exchange.is_pair_tradable(pair) and exchange.quote_price(pair) <= self.limit_price
