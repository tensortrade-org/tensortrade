# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import uuid

from enum import Enum
from typing import Callable
from decimal import Decimal

from tensortrade.base import TimedIdentifiable
from tensortrade.base.exceptions import InvalidOrderQuantity
from tensortrade.instruments import Quantity, ExchangePair
from tensortrade.orders import Trade, TradeSide, TradeType


class OrderStatus(Enum):
    PENDING = 'pending'
    OPEN = 'open'
    CANCELLED = 'cancelled'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'

    def __str__(self):
        return self.value


class Order(TimedIdentifiable):
    """
    Responsibilities of the Order:
        1. Confirming its own validity.
        2. Tracking its trades and reporting it back to the broker.
        3. Managing movement of quantities from order to order.
        4. Generating the next order in its path given that there is a
           'OrderSpec' for how to make the next order.
        5. Managing its own state changes when it can.
    """

    def __init__(self,
                 step: int,
                 side: TradeSide,
                 trade_type: TradeType,
                 exchange_pair: 'ExchangePair',
                 quantity: 'Quantity',
                 portfolio: 'Portfolio',
                 price: float,
                 criteria: Callable[['Order', 'Exchange'], bool] = None,
                 path_id: str = None,
                 start: int = None,
                 end: int = None):
        super().__init__()

        quantity = quantity.contain(exchange_pair)

        if quantity.size == 0:
            raise InvalidOrderQuantity(quantity)

        self.step = step
        self.side = side
        self.type = trade_type
        self.exchange_pair = exchange_pair
        self.portfolio = portfolio
        self.price = price
        self.criteria = criteria
        self.path_id = path_id or str(uuid.uuid4())
        self.quantity = quantity
        self.start = start or step
        self.end = end
        self.status = OrderStatus.PENDING

        self._specs = []
        self._listeners = []
        self._trades = []

        wallet = portfolio.get_wallet(
            self.exchange_pair.exchange.id,
            self.side.instrument(self.exchange_pair.pair)
        )

        if self.path_id not in wallet.locked.keys():
            self.quantity = wallet.lock(quantity, self, "LOCK FOR ORDER")

        self.remaining = self.quantity

    @property
    def size(self) -> Decimal:
        if not self.quantity or self.quantity is None:
            return -1
        return self.quantity.size

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, price: float):
        self._price = price

    @property
    def pair(self):
        return self.exchange_pair.pair

    @property
    def base_instrument(self) -> 'Instrument':
        return self.exchange_pair.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.exchange_pair.pair.quote

    @property
    def trades(self):
        return self._trades

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

    def is_executable(self):
        is_satisfied = self.criteria is None or self.criteria(self, self.exchange_pair.exchange)
        clock = self.exchange_pair.exchange.clock
        return is_satisfied and clock.step >= self.start

    def is_expired(self):
        if self.end:
            return self.exchange_pair.exchange.clock.step >= self.end
        return False

    def is_cancelled(self):
        return self.status == OrderStatus.CANCELLED

    def is_active(self):
        return self.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]

    def is_complete(self):
        if self.status == OrderStatus.CANCELLED:
            return True

        wallet = self.portfolio.get_wallet(
            self.exchange_pair.exchange.id,
            self.side.instrument(self.exchange_pair.pair)
        )
        quantity = wallet.locked.get(self.path_id, None)

        return (quantity and quantity.size == 0) or self.remaining.size <= 0

    def add_order_spec(self, order_spec: 'OrderSpec') -> 'Order':
        self._specs += [order_spec]
        return self

    def attach(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def detach(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self):
        self.status = OrderStatus.OPEN

        if self.portfolio.order_listener:
            self.attach(self.portfolio.order_listener)

        for listener in self._listeners or []:
            listener.on_execute(self)

        self.exchange_pair.exchange.execute_order(self, self.portfolio)

    def fill(self, trade: Trade):
        self.status = OrderStatus.PARTIALLY_FILLED

        filled = trade.quantity + trade.commission

        self.remaining -= filled
        self._trades += [trade]

        for listener in self._listeners or []:
            listener.on_fill(self, trade)

    def complete(self) -> 'Order':
        self.status = OrderStatus.FILLED

        order = None

        if self._specs:
            order_spec = self._specs.pop()
            order = order_spec.create_order(self)

        for listener in self._listeners or []:
            listener.on_complete(self)

        self._listeners = []

        return order or self.release("COMPLETED")

    def cancel(self, reason: str = "CANCELLED"):
        self.status = OrderStatus.CANCELLED

        for listener in self._listeners or []:
            listener.on_cancel(self)

        self._listeners = []

        self.release(reason)

    def release(self, reason: str = "RELEASE (NO REASON)"):
        for wallet in self.portfolio.wallets:
            if self.path_id in wallet.locked.keys():
                quantity = wallet.locked[self.path_id]

                if quantity is not None:
                    wallet.unlock(quantity, reason)

                wallet.locked.pop(self.path_id, None)

    def to_dict(self):
        return {
            "id": self.id,
            "step": self.step,
            "exchange_pair": str(self.exchange_pair),
            "status": self.status,
            "type": self.type,
            "side": self.side,
            "quantity": self.quantity,
            "size": self.size,
            "remaining": self.remaining,
            "price": self.price,
            "criteria": self.criteria,
            "path_id": self.path_id,
            "created_at": self.created_at
        }

    def to_json(self):
        return {
            "id": str(self.id),
            "step": int(self.step),
            "exchange_pair": str(self.exchange_pair),
            "status": str(self.status),
            "type": str(self.type),
            "side": str(self.side),
            "base_symbol": str(self.exchange_pair.pair.base.symbol),
            "quote_symbol": str(self.exchange_pair.pair.quote.symbol),
            "quantity": str(self.quantity),
            "size": float(self.size),
            "remaining": str(self.remaining),
            "price": float(self.price),
            "criteria": str(self.criteria),
            "path_id": str(self.path_id),
            "created_at": str(self.created_at)
        }

    def __iadd__(self, recipe: 'OrderSpec') -> 'Order':
        return self.add_order_spec(recipe)

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
