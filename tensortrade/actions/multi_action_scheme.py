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
from tensortrade.orders import Order
from tensortrade.instruments import Quantity
from tensortrade.trades import TradeSide


class MultiActionScheme(ActionScheme):
    """A discrete action scheme that determines actions based on a list of trading pairs, order criteria, and tradeable amounts."""

    registered_name = "actions"

    def __init__(self,
                 trading_pairs: Union[List['TradingPair'], 'TradingPair'],
                 order_criteria: Union[List['OrderCriteria'], 'OrderCriteria'],
                 tradeable_amounts: Union[List[float], int] = 10,
                 order_listener: 'OrderListener' = None):
        """
        Arguments:
            trading_pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            order_criteria: A list of criteria to select from when submitting an order.
            (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
            tradeable_amounts: A list of amounts to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
        """
        self._trading_pairs = self.context.get('trading_pairs', None) or trading_pairs if isinstance(
            trading_pairs, list) else [trading_pairs]
        self._order_criteria = self.context.get('order_criteria', None) or order_criteria if isinstance(
            order_criteria, list) else [order_criteria]
        self._tradeable_amounts = self.context.get('tradeable_amounts', None) or tradeable_amounts if isinstance(
            tradeable_amounts, list) else [1 / (x + 1) for x in range(tradeable_amounts)]
        self._order_listener = self.context.get('order_listener', None) or order_listener

        self.reset()

    @property
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        return self._action_space

    @property
    def trading_pairs(self) -> List['TradingPair']:
        """A list of trading pairs to select from when submitting an order..
        (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
        """
        return self._trading_pairs

    @trading_pairs.setter
    def trading_pairs(self, trading_pairs: Union[List['TradingPair'], 'TradingPair']):
        self._trading_pairs = trading_pairs if isinstance(
            trading_pairs, list) else [trading_pairs]

        self.reset()

    @property
    def order_criteria(self) -> List['OrderCriteria']:
        """A list of criteria to select from when submitting an order.
        (e.g. MarketOrderCriteria, LimitOrderCriteria, StopLossCriteria, CustomCriteria, etc.)
        """
        return self._order_criteria

    @order_criteria.setter
    def order_criteria(self, order_criteria: Union[List['OrderCriteria'], 'OrderCriteria']):
        self._order_criteria = order_criteria if isinstance(
            order_criteria, list) else [order_criteria]

        self.reset()

    @property
    def tradeable_amounts(self) -> List[float]:
        """A list of amounts to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
        """
        return self._tradeable_amounts

    @tradeable_amounts.setter
    def tradeable_amounts(self, tradeable_amounts: Union[List[float], int]):
        self._tradeable_amounts = tradeable_amounts if isinstance(tradeable_amounts, list) else [
            1 / (x + 1) for x in range(tradeable_amounts)]

        self.reset()

    def reset(self):
        self._actions = []

        for trading_pair, criteria, amount in product(self._trading_pairs, self._order_criteria, self._tradeable_amounts):
            self._actions += [(TradeSide.BUY, trading_pair, criteria, amount)]
            self._actions += [(TradeSide.SELL, trading_pair, criteria, amount)]

        self._action_space = Discrete(len(self._actions))

    @abstractmethod
    def get_order(self, action: int) -> Order:
        """Get the order to be executed on the exchange based on the action provided.

        Arguments:
            action: The action to be converted into an order.

        Returns:
            The order to be executed on the exchange this timestep.
        """
        (side, trading_pair, criteria, amount) = self._actions[action]

        quantity = Quantity(instrument=trading_pair.base_instrument, amount=amount)
        order = Order(side=side, pair=trading_pair, quantity=quantity, criteria=criteria)

        if self._order_listener is not None:
            order.add_listener(self._order_listener)

        return order
