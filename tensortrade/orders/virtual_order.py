
import uuid

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Callable


class OrderStatus(Enum):
    PENDING = 0
    OPEN = 1
    CANCELLED = 2
    FILLED = 3


class OrderListener(object, metaclass=ABCMeta):
    @abstractmethod
    def order_executed(self, order: 'VirtualOrder', exchange: 'Exchange'):
        raise NotImplementedError()

    @abstractmethod
    def order_cancelled(self, order: 'VirtualOrder', exchange: 'Exchange'):
        raise NotImplementedError()

    @abstractmethod
    def order_filled(self, order: 'VirtualOrder', exchange: 'Exchange'):
        raise NotImplementedError()


class VirtualOrder:
    def __init__(self, pair: 'TradingPair', quantity: 'Quantity', criteria: 'OrderCriteria', listener: 'OrderListener' = None):
        self.pair = pair
        self.quantity = quantity
        self.criteria = criteria
        self.listener = listener

        self.id = uuid.uuid4()
        self.status = OrderStatus.PENDING

    def is_executable(self, exchange: 'Exchange'):
        return self.criteria.is_executable(self.pair, exchange)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        if self.listener is not None:
            self.listener.order_executed(self, exchange)

        return exchange.execute_order(self)

    def on_fill(self, exchange: 'Exchange', amount: float):
        self.status = OrderStatus.FILLED

        if self.listener is not None:
            self.listener.order_filled(self, exchange, amount)

    def on_cancel(self):
        self.status = OrderStatus.CANCELLED

        if self.listener is not None:
            self.listener.order_cancelled(self)
