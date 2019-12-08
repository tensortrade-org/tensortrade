
from enum import Enum

from tensortrade.base import Identifiable
from tensortrade.trades import Trade, TradeSide, TradeType


class OrderStatus(Enum):
    PENDING = 0
    OPEN = 1
    CANCELLED = 2
    PARTIALLY_FILLED = 3
    FILLED = 4


class Order(Identifiable):

    def __init__(self,
                 side: TradeSide,
                 trade_type: TradeType,
                 pair: 'TradingPair',
                 quantity: 'Quantity',
                 price: float = None,
                 criteria: 'OrderCriteria' = None):
        self.side = side
        self.type = trade_type
        self.pair = pair
        self.quantity = quantity
        self.price = price
        self.criteria = criteria

        self.status = OrderStatus.PENDING

        self.quantity.lock_for(self.id)

        self._listeners = []

    @property
    def base_instrument(self) -> 'Instrument':
        return self.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.pair.quote

    @property
    def size(self) -> float:
        return self.quantity.amount

    @property
    def is_buy(self) -> bool:
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == TradeSide.SELL

    @property
    def is_limit_order(self) -> bool:
        return self.type == TradeType.LIMIT

    @property
    def is_market_order(self) -> bool:
        return self.type == TradeType.MARKET

    def is_executable(self, exchange: 'Exchange'):
        return self.criteria is None or self.criteria.is_satisfied(self, exchange)

    def attach(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def detach(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        if self._listeners:
            [listener.on_execute(self, exchange) for listener in self._listeners]

        return exchange.execute_order(self)

    def fill(self, exchange: 'Exchange', trade: Trade):
        self.status = OrderStatus.PARTIALLY_FILLED

        for listener in self._listeners or []:
            listener.on_fill(self, exchange, trade)

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
