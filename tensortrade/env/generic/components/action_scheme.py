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

from abc import abstractmethod, ABCMeta
from typing import Any

from gym.spaces import Space


from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed


class ActionScheme(Component, TimeIndexed, metaclass=ABCMeta):
    """A component for determining the action to take at each step of an
    episode.
    """

    registered_name = "actions"

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()

    @abstractmethod
    def perform(self, env: 'TradingEnv', action: Any) -> None:
        """Performs an action on the environment.

        Parameters
        ----------
        env : `TradingEnv`
            The trading environment to perform the `action` on.
        action : Any
            The action to perform on `env`.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        pass
