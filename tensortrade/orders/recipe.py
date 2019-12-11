

from enum import Enum
from typing import Callable, Union, Tuple, List

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InvalidOrderQuantity
from tensortrade.trades import Trade, TradeSide, TradeType
from .order import Order


class Recipe:
    def __init__(self,
                 side: TradeSide,
                 trade_type: TradeType,
                 pair: 'TradingPair',
                 criteria: Callable[['Order', 'Exchange'], bool] = None):
        self.side = side
        self.type = trade_type
        self.pair = pair
        self.criteria = criteria

    def create(self, order: 'Order', exchange: 'Exchange') -> 'Order':
        instrument = self.pair.base if self.side == TradeSide.BUY else self.pair.quote
        wallet = order.portfolio.get_wallet(exchange.id, instrument=instrument)
        size = wallet.locked[order.path_id]

        return Order(side=self.side,
                     trade_type=self.type,
                     pair=self.pair,
                     size=size,
                     portfolio=order.portfolio,
                     criteria=self.criteria)

    @staticmethod
    def validate(order: 'Order', recipe: 'Recipe'):
        pass
