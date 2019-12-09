

from typing import List, Union

from tensortrade.orders.criteria import OrderCriteria
from tensortrade.trades import TradeSide


class StopLossCriteria(OrderCriteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price."""

    def __init__(self, direction: str, percent: float):
        self.direction = direction
        self.percent = percent

    def is_satisfied(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradable(order.pair):
            return False

        price = exchange.quote_price(order.pair)
        percent = 100 * abs(price - order.price) / price

        above_satisfied = (price >= order.price and self.direction ==
                           "above" and percent >= self.percent)
        below_satisfied = (price <= order.price and self.direction ==
                           "below" and percent >= self.percent)

        return above_satisfied or below_satisfied
