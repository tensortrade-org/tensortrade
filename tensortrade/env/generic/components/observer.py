# Copyright 2020 The TensorTrade Authors.
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


from abc import abstractmethod


import numpy as np

from gym.spaces import Space


from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed


class Observer(Component, TimeIndexed):
    """A component to generate an observation at each step of an episode.
    """

    registered_name = "observer"

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self, env: 'TradingEnv') -> np.array:
        """Gets the observation at the current step of an episode

        Parameters
        ----------
        env: 'TradingEnv'
            The trading environment.

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the observer."""
        pass
