
import uuid

from enum import Enum
from typing import Callable, Union, Tuple, List

from tensortrade.base import TimedIdentifiable
from tensortrade.base.exceptions import InvalidOrderQuantity
from tensortrade.trades import Trade, TradeSide, TradeType


class OrderStatus(Enum):
    PENDING = 0
    OPEN = 1
    CANCELLED = 2
    PARTIALLY_FILLED = 3
    FILLED = 4

    def __str__(self):
        return str(self.value)


class Order(TimedIdentifiable):
    """
    Responsibilities of the Order:
        1. Confirming its own validity.
        2. Tracking its trades and for reporting it back to the broker.
        3. Managing movement of quantities from order to order.
        4. Generating the next order in its path given that there is a
           'Recipe' for how to make the next order.
        5. Managing its own state changes when it can.
    """

    def __init__(self,
                 side: TradeSide,
                 trade_type: TradeType,
                 pair: 'TradingPair',
                 quantity: 'Quantity',
                 portfolio: 'Portfolio',
                 price: float,
                 criteria: Callable[['Order', 'Exchange'], bool] = None,
                 path_id: str = None):
        if quantity.size == 0:
            raise InvalidOrderQuantity(quantity)

        self.side = side
        self.type = trade_type
        self.pair = pair
        self.quantity = quantity
        self.portfolio = portfolio
        self.price = price
        self.criteria = criteria
        self.path_id = path_id or self.id
        self.status = OrderStatus.PENDING

        self.filled_size = 0
        self.remaining_size = self.size

        self._recipes = []
        self._listeners = []
        self._trades = []

        self.quantity.lock_for(self.path_id)

    @property
    def size(self) -> float:
        size = self.quantity.size if self.pair.base is self.quantity.instrument else self.quantity.size * self.price
        return round(size, self.pair.base.precision)

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, price: float):
        self._price = round(price, self.pair.base.precision)

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
        return self.criteria is None or self.criteria(self, exchange)

    def is_complete(self):
        return self.remaining_size == 0

    def add_recipe(self, recipe: 'Recipe') -> 'Order':
        self._recipes = [recipe] + self._recipes
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
            wallet -= self.size * instrument
            wallet += self.quantity

        if self.portfolio.order_listener:
            self.attach(self.portfolio.order_listener)

        for listener in self._listeners or []:
            listener.on_execute(self, exchange)

        return exchange.execute_order(self, self.portfolio)

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

        if self._recipes:
            recipe = self._recipes.pop()
            order = recipe.create_order(self, exchange)

        for listener in self._listeners or []:
            listener.on_complete(self, exchange)

        self._listeners = []

        print('Completed: ', self.id, order)

        return order or self.release()

    def cancel(self, exchange: 'Exchange'):
        self.status = OrderStatus.CANCELLED

        for listener in self._listeners or []:
            listener.on_cancel(self, exchange)

        self._listeners = []
        self.release()

    def release(self):
        for wallet in self.portfolio.wallets:
            wallet.deallocate(self.path_id)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "type": self.type,
            "side": self.side,
            "pair": self.pair,
            "quantity": self.quantity,
            "size": self.size,
            "price": self.price,
            "criteria": self.criteria,
            "path_id": self.path_id
        }

    def to_json(self):
        return {
            "id": str(self.id),
            "status": str(self.status),
            "type": str(self.type),
            "side": str(self.side),
            "pair": str(self.pair),
            "quantity": str(self.quantity),
            "size": str(self.size),
            "price": str(self.price),
            "criteria": str(self.criteria),
            "path_id": str(self.path_id)
        }

    def __iadd__(self, recipe: 'Recipe') -> 'Order':
        return self.add_recipe(recipe)

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        recipes = [str(recipe) for recipe in self._recipes]
        return '<{}: {} | Recipes: {}>'.format(self.__class__.__name__, ', '.join(data), ', '.join(recipes))

    def __repr__(self):
        return str(self)
