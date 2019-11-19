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

import pandas as pd

from abc import abstractmethod

from tensortrade import Component
from tensortrade.trades import Trade


class RewardScheme(Component):

    registered_name = "rewards"

    def __init__(self):
        pass

    @property
    def exchange(self) -> 'Exchange':
        """The exchange being used by the current trading environments. Setting the exchange causes the scheme to reset."""
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'Exchange'):
        self._exchange = exchange
        self.reset()

    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass

    @abstractmethod
    def get_reward(self, current_step: int, trade: Trade) -> float:
        """
        Arguments:
            current_step: The environments's current timestep.
            trade: The trade executed and filled this timestep.

        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()
