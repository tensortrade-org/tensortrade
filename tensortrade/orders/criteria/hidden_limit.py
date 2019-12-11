
from typing import List, Union

from tensortrade.orders.criteria import Criteria
from tensortrade.trades import TradeSide


class HiddenLimit(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price, hidden from the public order book."""

    def __init__(self, limit_price: float):
        self.limit_price = limit_price

    def __call__(self, order: 'Order', exchange: 'Exchange') -> bool:
        if not exchange.is_pair_tradeable(order.pair):
            return False

        price = exchange.quote_price(order.pair)

        buy_satisfied = (order.side == TradeSide.BUY and price <= self.limit_price)
        sell_satisfied = (order.side == TradeSide.SELL and price >= self.limit_price)

        return buy_satisfied or sell_satisfied

    def __str__(self):
        return 'Limit: {}'.format(self.limit_price)

    def __repr__(self):
        return str(self)
