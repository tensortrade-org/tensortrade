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
from tensortrade.trades import Trade


class ActionScheme(Component):
    """A discrete action scheme for determining the action to take at each timestep within a trading environments."""

    registered_name = "actions"

    @abstractmethod
    def __init__(self,
                 instruments: Union[List['Instrument'], 'Instrument'],
                 trade_criteria: Union[List['TradeCriteria'], 'TradeCriteria'],
                 amount_splits: Union[List[float], int] = 10):
        """
        Arguments:
            instruments: The valid instruments to be traded by the agent.
            trade_criteria: The valid trade criteria required to submit an order.
            amount_splits: The number of times to split balances to determine valid trade amounts.
                (e.g. 4 results in splits of [1, 1/2, 1/3, 1/4], or you can pass in a custom list such as [1/3, 1/5, 1/7].)
            dtype: A type or str corresponding to the dtype of the `action_space`. Defaults to `np.float32`.
        """
        self._instruments = self.context.get('instruments', None) or list(instruments)
        self._trade_criteria = self.context.get('trade_criteria', None) or list(trade_criteria)
        self._amount_splits = self.context.get('amount_splits', None) or list(amount_splits)

        self.reset()

    @property
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        return self._action_space

    @property
    def instruments(self) -> List['Instrument']:
        """A type or str corresponding to the instruments of the `action_space`."""
        return self._instruments

    @instruments.setter
    def instruments(self, instruments: Union[List['Instrument'], 'Instrument']):
        self._instruments = instruments if isinstance(instruments, list) else [instruments]

        self.reset()

    @property
    def amount_splits(self) -> List[float]:
        """A type or str corresponding to the amount_splits of the `action_space`."""
        return self._amount_splits

    @amount_splits.setter
    def amount_splits(self, amount_splits: Union[List[float], int]):
        if isinstance(amount_splits, int):
            self._amount_splits = [float(1 / (x + 1)) for x in range(amount_splits)]
        else:
            self._amount_splits = amount_splits

        self.reset()

    @property
    def trade_criteria(self) -> List['TradeCriteria']:
        """A type or str corresponding to the trade_criteria of the `action_space`."""
        return self._trade_criteria

    @trade_criteria.setter
    def trade_criteria(self, trade_criteria: Union[List['TradeCriteria'], 'TradeCriteria']):
        self._trade_criteria = trade_criteria if isinstance(
            trade_criteria, list) else [trade_criteria]

        self.reset()

    def reset(self):
        self._open_orders = []
        self._actions = product(self._instruments, self._trade_criteria, self._amount_splits)
        self._action_space = Discrete(len(self._actions))

    @abstractmethod
    def get_trade(self, action: int) -> Trade:
        """Get the trade to be executed on the exchange based on the action provided.

        Arguments:
            action: The action to be converted into a trade.

        Returns:
            The trade to be executed on the exchange this timestep.
        """
        return self._actions[action]
