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


from abc import abstractmethod, ABCMeta
from typing import List, Tuple
from itertools import product
from gym.spaces import Discrete

from tensortrade import Component
from tensortrade.orders import Order


class ActionScheme(Component, metaclass=ABCMeta):
    """A discrete action scheme for determining the action to take at each timestep within a trading environments."""

    registered_name = "actions"

    def __init__(self):
        pass

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, actions):
        self._actions = actions

    def set_pairs(self, exchange_pairs: List[Tuple['Exchange', 'Pair']]):
        self.actions = list(product(exchange_pairs, self.actions))
        self.actions = [None] + self.actions

    def __len__(self) -> int:
        """The discrete action space produced by the action scheme."""
        return len(self.actions)

    def reset(self):
        """An optional reset method, which will be called each time the environment is reset."""
        pass

    def __add__(self, other):
        return AddActions(self, other)

    @abstractmethod
    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:
        """Get the order to be executed on the exchange based on the action provided.

        Arguments:
            action: The action to be converted into an order.
            exchange: The exchange the action will be executed on.
            portfolio: The portfolio of wallets used to execute the action.

        Returns:
            The order to be executed on the exchange this time step.
        """
        raise NotImplementedError()


class AddActions(ActionScheme):

    def __init__(self, left: ActionScheme, right: ActionScheme):
        super().__init__()
        self.left = left
        self.right = right

        self.actions = [None]

    def set_pairs(self, exchange_pairs):
        self.left.set_pairs(exchange_pairs)
        self.right.set_pairs(exchange_pairs)

        left_size = len(self.left.actions)
        right_size = len(self.right.actions)

        self.actions += [("left", i) for i in range(1, left_size)]
        self.actions += [("right", i) for i in range(1, right_size)]

    def get_order(self, action, portfolio):
        if action == 0:
            return None

        side, action = self.actions[action]

        if side == "left":
            order = self.left.get_order(action, portfolio)
            return order

        order = self.right.get_order(action, portfolio)

        return order
