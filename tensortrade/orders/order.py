
import uuid

from enum import Enum
from typing import Callable


class OrderStatus(Enum):

    PENDING = 0
    OPEN = 1
    CANCELLED = 2
    FILLED = 3


class OpenOrder:

    def __init__(self,
                 symbol: str,
                 side: str,
                 amount: 'Quantity',
                 criteria: Callable[['Exchange'], bool]):
        self.order_id = uuid.uuid1()
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.criteria = criteria

    def status(self) -> OrderStatus:
        raise NotImplemented()

    def is_executable(self, exchange: 'Exchange'):
        return self.criteria(exchange)

    def execute(self, exchange: 'Exchange') -> 'Trade':
        return exchange.execute_order(self)
