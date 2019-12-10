from typing import Callable, List

from tensortrade.base import Identifiable
from tensortrade.orders import Order, OrderListener, OrderStatus


class PathOrder(Identifiable):
    """An order that will be executed in multiple steps.
    The next step will execute each time the prior step is completed.
    """

    def __init__(self, orders: List[Order]):
        """
        Args:
            orders: A list of `Order`s to be executed in order of the list.
        """
        self._orders = orders

        for order in self._orders:
            order.assign(self.id)

        self._cursor = 0

    def __next__(self):
        if self._cursor >= len(self._orders):
            return None
        order = self._orders[self._cursor]
        self._cursor += 1
        return order
