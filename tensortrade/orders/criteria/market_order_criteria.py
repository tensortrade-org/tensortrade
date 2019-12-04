
from typing import List, Union

from .order_criteria import OrderCriteria


class MarketOrderCriteria(OrderCriteria):
    """An order criteria that allows execution immediately, at the current market price."""

    def __init__(self, enabled_pairs: Union[List['TradingPair'], 'TradingPair']):
        super().__init__(enabled_pairs=enabled_pairs)

    def is_executable(self, pair: 'TradingPair', exchange: 'Exchange') -> bool:
        return self.is_pair_enabled(pair) and exchange.is_pair_tradeable(pair)
