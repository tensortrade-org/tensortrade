

from typing import List, Union
from enum import Enum

from tensortrade.orders.criteria import OrderCriteria
from tensortrade.trades import TradeSide


class StopDirection(Enum):
    UP = 'up'
    DOWN = 'down'


class StopLossCriteria(OrderCriteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price."""

    def __init__(self, direction: StopDirection = StopDirection.DOWN, percent: float = 0.02):
        self.direction = direction
        self.percent = percent

    def is_satisfied(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradeable(order.pair):
            return False

        price = exchange.quote_price(order.pair)
        percent = abs(price - order.price) / price

        above_satisfied = (price >= order.price and self.direction ==
                           StopDirection.UP and percent >= self.percent)
        below_satisfied = (price <= order.price and self.direction ==
                           StopDirection.DOWN and percent >= self.percent)

        return above_satisfied or below_satisfied

    def __str__(self):
        return '{}: {}'.format('Take Profit' if self.direction == StopDirection.UP else 'Stop Loss', self.percent)

    def __repr__(self):
        return str(self)
