

from typing import Union
from enum import Enum

from tensortrade.orders.criteria import Criteria


class StopDirection(Enum):
    UP = 'up'
    DOWN = 'down'

    def __str__(self):
        return str(self.value)


class Stop(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is above or below a specific price."""

    def __init__(self, direction: Union[StopDirection, str], percent: float):
        self.direction = StopDirection(direction)
        self.percent = percent

    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        price = exchange.quote_price(order.pair)
        percent = abs(price - order.price) / price

        is_take_profit = (self.direction == StopDirection.UP) and (price >= order.price)
        is_stop_loss = (self.direction == StopDirection.DOWN) and (price <= order.price)

        return (is_take_profit or is_stop_loss) and percent >= self.percent

    def __str__(self):
        return '<Stop: direction={0}, percent={1}>'.format(self.direction,
                                                           self.percent)
