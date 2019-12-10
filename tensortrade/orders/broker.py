
from itertools import product
from typing import Union, List, Dict

from .order import Order, OrderStatus
from .order_listener import OrderListener
from .path_order import PathOrder


class Broker(OrderListener):
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
    def pending(self):
        return self._pending

    @property
    def unexecuted(self) -> List[Order]:
        """The list of orders the broker is waiting to execute, when their criteria is satisfied."""
        return self._unexecuted

    @property
    def executed(self) -> Dict[str, Order]:
        """The dictionary of orders the broker has executed since resetting, organized by order id"""
        return self._executed

    @property
    def trades(self) -> Dict[str, 'Trade']:
        """The dictionary of trades the broker has executed since resetting, organized by order id."""
        return self._trades

    def submit(self, order: Union[Order, PathOrder]):
        path_order = order.as_path() if isinstance(order, Order) else order
        order = next(path_order)
        if order:
            self._unexecuted += [order]
        self._pending[path_order.id] = path_order

    def cancel(self, order: Order):
        if order.status == OrderStatus.CANCELLED:
            raise Warning(
                'Cannot cancel order {} - order has already been cancelled.'.format(order.id))

        if order.status != OrderStatus.PENDING:
            raise Warning(
                'Cannot cancel order {} - order has already been executed.'.format(order.id))

        self._unexecuted.remove(order)

        order.cancel()

    def update(self):
        for order, exchange in product(self._unexecuted, self._exchanges):
            if order.is_executable(exchange):
                order.attach(self)
                order.execute(exchange)

                self._unexecuted.remove(order)
                self._executed[order.id] = order

    def on_fill(self, order: Order, exchange: 'Exchange', trade: 'Trade'):
        if trade.order_id in self._executed.keys() and trade not in self._trades:
            self._trades[trade.order_id] = self._trades[trade.order_id] or []
            self._trades[trade.order_id] += [trade]

            condition = lambda x: x.exchange_id == exchange.id
            trades_on_exchange = filter(condition, self._trades[trade.order_id])

            total_traded = sum([trade.size for trade in trades_on_exchange])

            if total_traded >= order.size:
                order.complete(exchange)

                # Generate next order or delete the path order
                # if all orders have been generated.
                path_order = self._pending[order.path_id]
                order = next(path_order)
                if order:
                    self._unexecuted += [order]
                else:
                    self._pending.pop(path_order.id)




    def reset(self):
        self._pending = {}
        self._unexecuted = []
        self._executed = {}
        self._trades = {}
