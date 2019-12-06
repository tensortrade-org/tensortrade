
from itertools import product
from typing import Union, List, Dict

from .order import OrderStatus


class Broker:
    """A broker for handling the execution of orders on multiple exchanges.
    Orders are kept in a virtual order book until they are ready to be executed.
    """

    def __init__(self, exchanges: Union[List['Exchange'], 'Exchange']):
        self._exchanges = exchanges if isinstance(exchanges, list) else [exchanges]

        self.reset()

    @property
    def exchanges(self) -> List['Exchange']:
        """The list of exchanges the broker will execute orders on."""
        return self._exchanges

    @exchanges.setter
    def exchanges(self, exchanges: Union[List['Exchange'], 'Exchange']):
        self._exchanges = exchanges if isinstance(exchanges, list) else [exchanges]

    @property
    def unexecuted_orders(self) -> List['Order']:
        """The list of orders the broker is waiting to execute, when their criteria passes."""
        return self._unexecuted_orders

    @property
    def executed_orders(self) -> Dict[str, 'Order']:
        """The list of orders the broker has executed since resetting."""
        return self._executed_orders

    def create_order(self, order: 'Order'):
        self._unexecuted_orders += [order]

    def cancel(self, order: 'Order'):
        if order.status == OrderStatus.CANCELLED:
            raise Warning(
                'Cannot cancel order {} - order has already been cancelled.'.format(order.id))

        if order.status != OrderStatus.PENDING:
            raise Warning(
                'Cannot cancel order {} - order has already been executed.'.format(order.id))

        self._unexecuted_orders.remove(order)

        order.on_cancel()

    def update(self):
        for order, exchange in product(self._unexecuted_orders, self._exchanges):
            if order.is_executable(exchange):
                order.execute(exchange)

                self._unexecuted_orders.remove(order)
                self._executed_orders[order.id] = order

        for exchange in self.exchanges:
            for trade in exchange.trades:
                if trade.order_id in self._executed_orders.keys() and trade not in self._trades:
                    self._trades += [trade]
                    self._executed_orders[trade.order_id].on_fill(exchange, trade.amount)

    def reset(self):
        self._trades = []
        self._unexecuted_orders = []
        self._executed_orders = {}
