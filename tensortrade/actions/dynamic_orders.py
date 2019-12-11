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

import numpy as np

from typing import Union, List
from abc import abstractmethod
from itertools import product
from gym.spaces import Discrete

from tensortrade.actions import ActionScheme
from tensortrade.trades import TradeSide, TradeType
from tensortrade.instruments import Quantity
from tensortrade.orders import Order, OrderListener


class DynamicOrders(ActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes."""

    def __init__(self,
                 pairs: Union[List['TradingPair'], 'TradingPair'],
                 criteria: Union[List['OrderCriteria'], 'OrderCriteria'] = None,
                 trade_sizes: Union[List[float], int] = 10,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):
        """
        Arguments:
            pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            criteria: A list of order criteria to select from when submitting an order.
            (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
            trade_sizes: A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        self.pairs = self.default('pairs', pairs)
        self.criteria = self.default('criteria', criteria)
        self.trade_sizes = self.default('trade_sizes', trade_sizes)
        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self.reset()

    @property
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        return Discrete(len(self._actions))

    @property
    def pairs(self) -> List['TradingPair']:
        """A list of trading pairs to select from when submitting an order.
        (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
        """
        return self._pairs

    @pairs.setter
    def pairs(self, pairs: Union[List['TradingPair'], 'TradingPair']):
        self._pairs = pairs if isinstance(pairs, list) else [pairs]

    @property
    def criteria(self) -> List['OrderCriteria']:
        """A list of order criteria to select from when submitting an order.
        (e.g. MarketOrderCriteria, LimitOrderCriteria, StopLossCriteria, CustomCriteria, etc.)
        """
        return self._criteria

    @criteria.setter
    def criteria(self, criteria: Union[List['OrderCriteria'], 'OrderCriteria']):
        self._criteria = criteria if isinstance(criteria, list) else [criteria]

    @property
    def trade_sizes(self) -> List[float]:
        """A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
        """
        return self._trade_sizes

    @trade_sizes.setter
    def trade_sizes(self, trade_sizes: Union[List[float], int]):
        self._trade_sizes = trade_sizes if isinstance(trade_sizes, list) else [
            (x + 1) / trade_sizes for x in range(trade_sizes)]

    def get_order(self, action: int, exchange: 'Exchange', portfolio: 'Portfolio') -> Order:
        if action == 0:
            return None

        (side, pair, criteria, size) = self._actions[action]

        instrument = pair.base if side == TradeSide.BUY else pair.quote
        wallet = portfolio.get_wallet(exchange.id, instrument=instrument)
        price = exchange.quote_price(instrument)
        size = min(wallet.balance.size, (wallet.balance.size * size))

        if size < 10 ** -instrument.precision:
            return None

        quantity = size * instrument

        wallet -= quantity

        order = Order(side=side,
                      trade_type=self._trade_type,
                      pair=pair,
                      price=price,
                      quantity=quantity,
                      portfolio=portfolio,
                      criteria=criteria)

        quantity.lock_for(order.id)

        wallet += quantity

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return order

    def reset(self):
        self._actions = [None]

        for trading_pair, criteria, size in product(self._pairs, self._criteria, self._trade_sizes):
            self._actions += [(TradeSide.BUY, trading_pair, criteria, size)]
            self._actions += [(TradeSide.SELL, trading_pair, criteria, size)]
