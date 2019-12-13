

from enum import Enum
from typing import Callable, Union, Tuple, List

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InvalidOrderQuantity
from tensortrade.trades import Trade, TradeSide, TradeType
from .order import Order


class Recipe(Identifiable):
    def __init__(self,
                 side: TradeSide,
                 trade_type: TradeType,
                 pair: 'TradingPair',
                 criteria: Callable[['Order', 'Exchange'], bool] = None):
        self.side = side
        self.type = trade_type
        self.pair = pair
        self.criteria = criteria

    def create_order(self, order: 'Order', exchange: 'Exchange') -> 'Order':
        base_instrument = self.pair.base if self.side == TradeSide.BUY else self.pair.quote
        wallet = order.portfolio.get_wallet(exchange.id, instrument=base_instrument)
        quantity = wallet.locked[order.path_id]

        return Order(side=self.side,
                     trade_type=self.type,
                     pair=self.pair,
                     quantity=quantity,
                     portfolio=order.portfolio,
                     price=order.price,
                     criteria=self.criteria,
                     path_id=order.path_id)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "pair": self.pair,
            "criteria": self.criteria
        }

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
