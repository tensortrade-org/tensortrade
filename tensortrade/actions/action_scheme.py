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
from gym.spaces import Space

from tensortrade import Component, TimeIndexed
from tensortrade.orders import Order


class ActionScheme(Component, TimeIndexed, metaclass=ABCMeta):
    """An action scheme for determining the action to take at each time step within a trading environment."""

    registered_name = "actions"

    def __init__(self):
        self._action_space = None
        self._exchange_pairs = None

    @property
    def exchange_pairs(self):
        return self._exchange_pairs

    @exchange_pairs.setter
    def exchange_pairs(self, exchange_pairs: List['ExchangePair']):
        self._exchange_pairs = exchange_pairs

    @property
    def action_space(self) -> Space:
        return self._action_space

    @abstractmethod
    def compile(self):
        raise NotImplementedError()

    @abstractmethod
    def get_order(self, action: any, portfolio: 'Portfolio') -> Order:
        """Get the order to be executed on the exchange based on the action provided.

        Arguments:
            action : any
                The action to be converted into an order.
            portfolio : 'Portfolio'
                The portfolio the environment is operating on.

        Returns:
            The order to be executed on the exchange this time step.
        """
        raise NotImplementedError()

    def reset(self):
        """An optional reset method, which will be called each time the environment is reset."""
        pass
