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

from tensortrade import Component
from tensortrade.orders import VirtualOrder


class ActionScheme(Component):
    """A discrete action scheme for determining the action to take at each timestep within a trading environments."""

    registered_name = "actions"

    @abstractmethod
    def __init__(self, order_criteria: Union[List['OrderCriteria'], 'OrderCriteria'], tradeable_amounts: Union[List[float], int] = 10):
        """
        Arguments:
            order_criteria: The criteria necessary to submit an order.
            tradeable_amounts: The list of amounts tradeable by this action scheme.
            (E.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
        """
        self._order_criteria = self.context.get('order_criteria', None) or order_criteria if isinstance(
            order_criteria, list) else[order_criteria]
        self._tradeable_amounts = self.context.get('tradeable_amounts', None) or tradeable_amounts if isinstance(
            tradeable_amounts, list) else [1 / (x + 1) for x in range(tradeable_amounts)]

        self.reset()

    @property
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        return self._action_space

    @property
    def order_criteria(self) -> List['OrderCriteria']:
        """A type or str corresponding to the order_criteria of the `action_space`."""
        return self._order_criteria

    @order_criteria.setter
    def order_criteria(self, order_criteria: Union[List['OrderCriteria'], 'OrderCriteria']):
        self._order_criteria = order_criteria if isinstance(
            order_criteria, list) else [order_criteria]

        self.reset()

    def reset(self):
        self._open_orders = []
        self._actions = []

        for criteria, amount in product(self._order_criteria.trade_pairs, self._tradeable_amounts):
            self._actions += [(criteria, amount)]

        self._action_space = Discrete(len(self._actions))

    @abstractmethod
    def get_order(self, action: int) -> VirtualOrder:
        """Get the order to be executed on the exchange based on the action provided.

        Arguments:
            action: The action to be converted into a trade.

        Returns:
            The trade to be executed on the exchange this timestep.
        """
        return self._actions[action]
