import uuid

from typing import Callable, List

from . import Order, OrderListener, OrderStatus


class MultiStepOrder(Order, OrderListener):
    def __init__(self, steps: List['Order'], listener: 'OrderListener' = None):
        self.steps = steps

        self.id = uuid.uuid4()
        self.status = OrderStatus.PENDING

        self._active_step = 0
        self._active_order = steps[self._active_step]
        self._listeners = []

    def is_executable(self, exchange: 'Exchange'):
        return self._active_order.criteria.is_executable(self._active_order, exchange)

    def add_listener(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def remove_listener(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        if self._listeners:
            [listener.order_executed(self, exchange) for listener in self._listeners]

        return self._active_order.execute(exchange)

    def on_fill(self, exchange: 'Exchange', amount: float):
        self.status = OrderStatus.PARTIALLY_FILLED

        if self._listeners:
            [listener.order_filled(self, exchange, amount) for listener in self._listeners]

    def on_complete(self, exchange: 'Exchange'):
        self.status = OrderStatus.FILLED

        if self._listeners:
            [listener.order_completed(self, exchange) for listener in self._listeners]

        self._listeners = []

    def on_cancel(self):
        self.status = OrderStatus.CANCELLED

        if self._listeners:
            [listener.order_cancelled(self) for listener in self._listeners]

        self._listeners = []

    def order_cancelled(self, order: 'Order'):
        self.on_cancel()

    def order_filled(self, order: 'Order', exchange: 'Exchange', amount: float):
        self.on_fill(exchange, amount)

    def order_completed(self, order: 'Order', exchange: 'Exchange'):
        self._active_step += 1
        self._active_order = self.steps[self._active_step]

        if self._active_step == len(self.steps):
            self.on_complete(exchange)
