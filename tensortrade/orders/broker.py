
from itertools import product
from enum import Enum
from typing import Callable, List

from .order import OrderStatus


class Broker:

    def __init__(self, exchanges: List['Exchange'], action_scheme: 'ActionScheme'):
        self.exchanges = exchanges
        self.order_history = []
        self.order_book = []

    def send(self, order: 'Order'):
        self.order_book += [order]

    def cancel(self, order: 'Order'):
        self.order_book.remove(order)

    def update(self) -> List['Trade']:
        trades = []
        for order, exchange in product(self.order_book, self.exchanges):
            if order.is_executable(exchange):
                trade = order.execute(exchange)
                status = order.status()
                if status == OrderStatus.FILLED:
                    self.action_scheme.on_fill(trade)
                elif status == OrderStatus.CANCELLED:
                    self.action_scheme.on_cancel(trade)
                trades += [trade]
        return trades
