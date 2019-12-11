
import uuid

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
                 size: 'Quantity',
                 portfolio: 'Portfolio',
                 price: float = None,
                 criteria: Callable[['Order', 'Exchange'], bool] = None,
                 path_id: str = None):
        if size.amount == 0:
            raise InvalidOrderQuantity(size)

        self.side = side
        self.type = trade_type
        self.pair = pair
        self.size = size
        self.portfolio = portfolio
        self.price = price
        self.criteria = criteria
        self.recipes = []
        self.status = OrderStatus.PENDING

        if path_id and size.is_locked:
            if path_id != size.path_id:
                raise Exception("Path ID Mismatch: Size {} and Order".format(path_id, size.path_id))
        elif path_id and not size.is_locked:
            self.path_id = path_id
            self.size.lock_for(self.path_id)
        elif not path_id and size.is_locked:
            self.path_id = size.path_id
        else:
            self.path_id = str(uuid.uuid4())
            self.size.lock_for(self.path_id)

        self._listeners = []

        # Keep track of whether the order is filled or not.
        self.filled_size = 0
        self.remaining_size = self.size.amount
        self._trades = []

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

    def is_done(self):
        return self.remaining_size == 0

    def __iadd__(self, other):
        self.recipes = [other] + self.recipes
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
            print("Getting initial amounts form wallet.")
            wallet -= self.size.amount * instrument
            wallet += self.size

        for listener in self._listeners or []:
            listener.on_execute(self, exchange)

        return exchange.execute_order(self, self.portfolio)

    def fill(self, exchange: 'Exchange', trade: Trade):
        self.status = OrderStatus.PARTIALLY_FILLED

        self.filled_size += trade.size.amount
        self.remaining_size -= trade.size.amount

        for listener in self._listeners or []:
            listener.on_fill(self, exchange, trade)

    def complete(self, exchange: 'Exchange') -> 'Order':
        self.status = OrderStatus.FILLED

        # Create the next order to be returned if there is one.
        # The order created by this method should automatically
        # lock the quantity to be associated with the path_id that
        # it is currently on.
        order = None
        if len(self.recipes) > 0:
            recipe = self.recipes.pop()
            order = recipe.create(self, exchange)

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
            wallet.unlock(self.path_id)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "type": self.type,
            "pair": self.pair,
            "size": self.size,
            "price": self.price,
            "criteria": self.criteria
        }

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
