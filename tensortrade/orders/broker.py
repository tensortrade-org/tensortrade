
from itertools import product
from typing import Union, List, Dict

from tensortrade.base.core import TimeIndexed

from .order import Order, OrderStatus
from .order_listener import OrderListener


class Broker(OrderListener, TimeIndexed):
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

    def submit(self, order: Order):
        self._unexecuted += [order]

    def cancel(self, order: Order, exchange: 'Exchange'):
        if order.status == OrderStatus.CANCELLED:
            raise Warning(
                'Cannot cancel order {} - order has already been cancelled.'.format(order.id))

        if order.status != OrderStatus.PENDING:
            raise Warning(
                'Cannot cancel order {} - order has already been executed.'.format(order.id))

        self._unexecuted.remove(order)

        order.cancel(exchange)

    def update(self):
        for order, exchange in product(self._unexecuted, self._exchanges):
            if order.is_executable_on(exchange):
                print('Execute: ', order)

                self._unexecuted.remove(order)
                self._executed[order.id] = order

                order.attach(self)
                order.execute(exchange)

    def on_fill(self, order: Order, exchange: 'Exchange', trade: 'Trade'):
        print('Fill: ', trade)

        if trade.order_id in self._executed.keys() and trade not in self._trades:
            self._trades[trade.order_id] = self._trades.get(trade.order_id, [])
            self._trades[trade.order_id] += [trade]

            print('Total traded: ', order.filled_size)

            if order.is_complete():
                next_order = order.complete(exchange)

                if next_order:
                    self.submit(next_order)

    def reset(self):
        self._unexecuted = []
        self._executed = {}
        self._trades = {}
