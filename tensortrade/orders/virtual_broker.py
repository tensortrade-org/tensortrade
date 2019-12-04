
from itertools import product
from enum import Enum
from typing import Callable, List

from .virtual_order import OrderStatus


class VirtualBroker:
    """A broker for handling the execution of orders on multiple exchanges.

    Orders are kept in a virtual order book until they are ready to be executed.
    """

    def __init__(self, exchanges: List['Exchange']):
        self._exchanges = exchanges

        self.reset()

    @property
    def exchanges(self) -> List['Exchange']:
        return self._exchanges

    @exchanges.setter
    def exchanges(self, exchanges: List['Exchange']):
        self._exchanges = exchanges

    def send(self, order: 'Order'):
        self.unexecuted_orders += [order]

    def cancel(self, order: 'Order'):
        if order.status is not OrderStatus.PENDING:
            raise Warning(
                'Cannot cancel order {} - order has already been executed.'.format(order.id))

        self.unexecuted_orders.remove(order)

        order.on_cancel()

    def update(self):
        for order, exchange in product(self.unexecuted_orders, self.exchanges):
            if order.is_executable(exchange):
                order.execute(exchange)

                self.unexecuted_orders.remove(order)
                self.executed_orders[order.id] = order

        for exchange in self.exchanges:
            for trade in exchange.trades:
                if trade.order_id in self.executed_orders.keys():
                    self.executed_orders[trade.order_id].on_fill(exchange, trade.amount)

    def reset(self):
        self.unexecuted_orders = []
        self.executed_orders = {}
        self.executed_order_ids = []
