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


from enum import Enum
from typing import Callable

from tensortrade.base import TimedIdentifiable
from tensortrade.base.exceptions import InvalidOrderQuantity, InsufficientFunds
from tensortrade.instruments import Quantity, Price
from tensortrade.trades import Trade, TradeSide, TradeType


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
                 pair: 'TradingPair',
                 quantity: 'Quantity',
                 portfolio: 'Portfolio',
                 price: 'Price',
                 criteria: Callable[['Order', 'Exchange'], bool] = None,
                 path_id: str = None,
                 ttl_in_seconds: int = None,
                 ttl_in_steps: int = None):
        super().__init__()

        if quantity.size == 0:
            raise InvalidOrderQuantity(quantity)

        self.step = step
        self.side = side
        self.type = trade_type
        self.pair = pair
        self.quantity = quantity
        self.portfolio = portfolio
        self.price = price
        self.criteria = criteria
        self.path_id = path_id or self.id
        self.ttl_in_seconds = ttl_in_seconds
        self.ttl_in_steps = ttl_in_steps
        self.status = OrderStatus.PENDING

        self.filled_size = 0
        self.remaining_size = self.size

        self._specs = []
        self._listeners = []
        self._trades = []

        self.quantity.lock_for(self.path_id)

    @property
    def size(self) -> float:
        if self.pair.base is self.quantity.instrument:
            return self.quantity.size
        return self.quantity.size * self.price.rate

    @property
    def price(self) -> 'Price':
        return self._price

    @price.setter
    def price(self, price: 'Price'):
        self._price = price

    @property
    def base_instrument(self) -> 'Instrument':
        return self.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.pair.quote

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

    def is_executable_on(self, exchange: 'Exchange'):
        if not exchange.is_pair_tradable(self.pair):
            return False
        return self.criteria is None or self.criteria(self, exchange)

    def is_complete(self):
        return self.remaining_size == 0

    def add_order_spec(self, order_spec: 'OrderSpec') -> 'Order':
        self._specs = [order_spec] + self._specs
        return self

    def attach(self, listener: 'OrderListener'):
        self._listeners += [listener]

    def detach(self, listener: 'OrderListener'):
        self._listeners.remove(listener)

    def execute(self, exchange: 'Exchange'):
        self.status = OrderStatus.OPEN

        instrument = self.pair.base if self.side == TradeSide.BUY else self.pair.quote
        wallet = self.portfolio.get_wallet(exchange.id, instrument=instrument)

        if self.path_id not in wallet.locked.keys():
            try:
                wallet -= self.size * instrument
            except InsufficientFunds:
                size = wallet.balance.size
                wallet -= size * instrument
                self.quantity = Quantity(instrument, size, path_id=self.path_id)

            wallet += self.quantity

        if self.portfolio.order_listener:
            self.attach(self.portfolio.order_listener)

        for listener in self._listeners or []:
            listener.on_execute(self, exchange)

        exchange.execute_order(self, self.portfolio)

    def fill(self, exchange: 'Exchange', trade: Trade):
        self.status = OrderStatus.PARTIALLY_FILLED

        fill_size = round(trade.size + trade.commission.size, self.pair.base.precision)

        self.filled_size += fill_size
        self.remaining_size -= fill_size

        for listener in self._listeners or []:
            listener.on_fill(self, exchange, trade)

    def complete(self, exchange: 'Exchange') -> 'Order':
        self.status = OrderStatus.FILLED

        order = None

        if self._specs:
            order_spec = self._specs.pop()
            order = order_spec.create_order(self, exchange)

        for listener in self._listeners or []:
            listener.on_complete(self, exchange)

        self._listeners = []

        return order or self.release()

    def cancel(self):
        self.status = OrderStatus.CANCELLED

        for listener in self._listeners or []:
            listener.on_cancel(self)

        self._listeners = []
        self.release()

    def release(self):
        for wallet in self.portfolio.wallets:
            wallet.deallocate(self.path_id)

    def to_dict(self):
        return {
            "id": self.id,
            "step": self.step,
            "status": self.status,
            "type": self.type,
            "side": self.side,
            "pair": self.pair,
            "quantity": self.quantity,
            "size": self.size,
            "filled_size": self.filled_size,
            "price": self.price,
            "criteria": self.criteria,
            "path_id": self.path_id,
            "created_at": self.created_at
        }

    def to_json(self):
        return {
            "id": str(self.id),
            "step": int(self.step),
            "status": str(self.status),
            "type": str(self.type),
            "side": str(self.side),
            "base_symbol": str(self.pair.base.symbol),
            "quote_symbol": str(self.pair.quote.symbol),
            "quantity": str(self.quantity),
            "size": float(self.size),
            "filled_size": self.filled_size,
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
