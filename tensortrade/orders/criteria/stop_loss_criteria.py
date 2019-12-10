

from typing import List, Union
from enum import Enum

from tensortrade.orders.criteria import Criteria
from tensortrade.trades import TradeSide


class StopDirection(Enum):
    UP = 'up'
    DOWN = 'down'
    EITHER = 'either'


class StopLossCriteria(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is above or below a specific price."""

    def __init__(self, direction: StopDirection = StopDirection.DOWN, up_percent: float = 0.02, down_percent: float = 0.02, percent: float = None):
        self.direction = direction
        self.up_percent = up_percent
        self.down_percent = down_percent

        if percent:
            self.up_percent = percent
            self.down_percent = percent

    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradeable(order.pair):
            return False

        price = exchange.quote_price(order.pair)
        percent = abs(price - order.price) / price

        take_profit_satisfied = self.direction in [
            StopDirection.UP, StopDirection.EITHER] and price >= order.price and percent >= self.up_percent
        stop_loss_satisfied = self.direction in [
            StopDirection.DOWN, StopDirection.EITHER] and price <= order.price and percent >= self.down_percent

        if self.direction in [StopDirection.UP, StopDirection.EITHER]:
            print('Take profit waiting: ({}/{})'.format(percent, self.up_percent))

        if self.direction in [StopDirection.DOWN, StopDirection.EITHER]:
            print('Stop loss waiting: ({}/{})'.format(percent, self.down_percent))

        return take_profit_satisfied or stop_loss_satisfied

    def __str__(self):
        return 'StopLoss: {} | {} (T/P) | {} (S/L)'.format(self.direction, self.up_percent, self.down_percent)

    def __repr__(self):
        return str(self)
