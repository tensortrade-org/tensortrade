
import uuid

from enum import Enum

from ..trades import TradeSide


class OrderStatus(Enum):
    PENDING = 0
    OPEN = 1
    CANCELLED = 2
    PARTIALLY_FILLED = 3
    FILLED = 4


class Order:
    def __init__(self, side: TradeSide, pair: 'TradingPair', quantity: 'Quantity', criteria: 'OrderCriteria' = None):
        self.side = side
        self.pair = pair
        self.quantity = quantity
        self.criteria = criteria

        self.id = uuid.uuid4()
        self.status = OrderStatus.PENDING

        self.quantity.order_id = self.id

        self._listeners = []

    @property
    def is_buy(self) -> bool:
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == TradeSide.SELL

    @property
    def base_instrument(self) -> 'Instrument':
        return self.pair.base_instrument

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.pair.quote_instrument

    @property
    def amount(self) -> float:
        return self.quantity.amount

    def is_executable(self, exchange: 'Exchange'):
        return self.criteria is None or self.criteria.is_satisfied(self, exchange)

    def add_listener(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def remove_listener(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        if self._listeners:
            [listener.order_executed(self, exchange) for listener in self._listeners]

        return exchange.execute_order(self)

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
