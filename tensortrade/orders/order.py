
from enum import Enum
from typing import Callable, Union, Tuple, List

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InvalidOrderQuantity
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
                 portfolio: 'Portfolio',
                 price: float = None,
                 criteria: Callable[['Order', 'Exchange'], bool] = None,
                 followed_by: 'Order' = None):
        if quantity.amount == 0:
            raise InvalidOrderQuantity(quantity)

        self.side = side
        self.type = trade_type
        self.pair = pair
        self.quantity = quantity
        self.portfolio = portfolio
        self.price = price
        self.criteria = criteria
        self.followed_by = followed_by
        self.status = OrderStatus.PENDING
        self.path_id = self.id

        if followed_by:
            self.follow_by(followed_by)

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

    def is_executable_on(self, exchange: 'Exchange'):
        return self.criteria is None or self.criteria(self, exchange)

    def follow_by(self, order: 'Order' = None):
        self.followed_by = order
        order.path_id = self.id

    def begin_path(self, orders: Union[List['Order'], 'Order'] = None):
        path = [self, *orders] if isinstance(orders, list) else [self, orders]

        for idx, order in enumerate(path):
            order.path_id = self.id

            if len(path) > idx:
                order.follow_by(path[idx + 1])

    def attach(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def detach(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        for listener in self._listeners or []:
            listener.on_execute(self, exchange)

        return exchange.execute_order(self, self.portfolio)

    def fill(self, exchange: 'Exchange', trade: Trade):
        self.status = OrderStatus.PARTIALLY_FILLED

        for listener in self._listeners or []:
            listener.on_fill(self, exchange, trade)

    def complete(self, exchange: 'Exchange'):
        self.status = OrderStatus.FILLED

        for listener in self._listeners or []:
            listener.on_complete(self, exchange)

        self._listeners = []
        self.release()

    def cancel(self):
        self.status = OrderStatus.CANCELLED

        for listener in self._listeners or []:
            listener.on_cancel(self)

        self._listeners = []
        self.release()

    def release(self):
        for wallet in self.portfolio.wallets:
            wallet.unlock(self.id)

    def __str__(self):
        return '{} | {} | {} | {} | {} | {} | {} | {} -> {}'.format(self.id,
                                                                    self.status,
                                                                    self.side,
                                                                    self.type,
                                                                    self.pair,
                                                                    self.quantity,
                                                                    self.price,
                                                                    self.criteria,
                                                                    self.followed_by)

    def __repr__(self):
        return str(self)
