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

from abc import abstractmethod, ABCMeta
from typing import Union, List
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
    @abstractmethod
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        raise NotImplementedError()

    def reset(self):
        """An optional reset method, which will be called each time the environment is reset."""
        pass

    @abstractmethod
    def get_order(self, action: int, exchange: 'Exchange', portfolio: 'Portfolio') -> Order:
        """Get the order to be executed on the exchange based on the action provided.

        Arguments:
            action: The action to be converted into an order.
            exchange: The exchange the action will be executed on.
            portfolio: The portfolio of wallets used to execute the action.

        Returns:
            The order to be executed on the exchange this timestep.
        """
        raise NotImplementedError()
