
from typing import List, Union

from .order_criteria import OrderCriteria
from ...trades import TradeSide


class LimitOrderCriteria(OrderCriteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price."""

    def __init__(self, limit_price: float):
        self.limit_price = limit_price

    def is_satisfied(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradable(order.pair):
            return False

        price = exchange.quote_price(order.pair)

        buy_satisfied = (order.side == TradeSide.BUY and price <= self.limit_price)
        sell_satisfied = (order.side == TradeSide.SELL and price >= self.limit_price)

        return buy_satisfied or sell_satisfied
