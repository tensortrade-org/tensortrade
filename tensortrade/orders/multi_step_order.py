import uuid

from typing import Callable, List

from . import Order, OrderListener, OrderStatus


class MultiStepOrder(OrderListener):
    def __init__(self, steps: List['Order'], listener: 'OrderListener' = None):
        self.steps = steps

        self.id = uuid.uuid4()
        self.status = OrderStatus.PENDING

        self._active_step = 0
        self._active_order = steps[self._active_step]
        self._listeners = []

    def is_executable(self, exchange: 'Exchange'):
        return self._active_order.criteria.is_satisfied(self._active_order, exchange)

    def attach(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def detach(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        if self._listeners:
            [listener.order_executed(self, exchange) for listener in self._listeners]

        return self._active_order.execute(exchange)

    def fill(self, exchange: 'Exchange', amount: float):
        self.status = OrderStatus.PARTIALLY_FILLED

        for listener in self._listeners or []:
            listener.on_fill(self, exchange, amount)

    def complete(self, exchange: 'Exchange'):
        self.status = OrderStatus.FILLED

        for listener in self._listeners or []:
            listener.on_complete(self, exchange)

        self._listeners = []

    def cancel(self):
        self.status = OrderStatus.CANCELLED

        for listener in self._listeners or []:
            listener.on_cancel(self)

        self._listeners = []

    def on_cancel(self, order: 'Order'):
        self.cancel()

    def on_fill(self, order: 'Order', exchange: 'Exchange', amount: float):
        self.fill(exchange, amount)

    def on_complete(self, order: 'Order', exchange: 'Exchange'):
        self._active_step += 1
        self._active_order = self.steps[self._active_step]

        if self._active_step == len(self.steps):
            self.complete(exchange)
