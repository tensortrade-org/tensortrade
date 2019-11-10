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

from typing import Union
from abc import abstractmethod
from gym.spaces import Space

from tensortrade import Component
from tensortrade.trades import Trade

DTypeString = Union[type, str]
TradeActionUnion = Union[int, float, tuple]


class ActionScheme(Component):
    """An abstract scheme for determining the action to take at each timestep within a trading environments."""

    registered_name = "actions"

    @abstractmethod
    def __init__(self, action_space: Space, dtype: DTypeString = np.float16):
        """
        Arguments:
            action_space: The shape of the actions produced by the scheme.
            dtype: A type or str corresponding to the dtype of the `action_space`. Defaults to `np.float16`.
        """
        self._action_space = action_space
        self._dtype = self.context.get('dtype', None) or dtype

    @property
    def action_space(self) -> Space:
        """The shape of the actions produced by the scheme."""
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: Space):
        self._action_space = action_space

    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        self._dtype = dtype

    @property
    def exchange(self) -> 'Exchange':
        """The exchange being used by the current trading environments.

        This will be set by the trading environments upon initialization. Setting the exchange causes the scheme to reset.
        """
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'Exchange'):
        self._exchange = exchange
        self.reset()

    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass

    @abstractmethod
    def get_trade(self, action: TradeActionUnion) -> Trade:
        """Get the trade to be executed on the exchange based on the action provided.

        Arguments:
            action: The action to be converted into a trade.

        Returns:
            The trade to be executed on the exchange this timestep.
        """
        raise NotImplementedError
