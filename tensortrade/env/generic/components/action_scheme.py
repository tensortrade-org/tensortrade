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

from gym.spaces import Space

from tensortrade.base.component import Component
from tensortrade.base.core import TimeIndexed


class ActionScheme(Component, TimeIndexed, metaclass=ABCMeta):
    """An action scheme for determining the action to take at each time step within a trading environment."""

    registered_name = "actions"

    @property
    @abstractmethod
    def action_space(self) -> Space:
        raise NotImplementedError()

    @abstractmethod
    def perform(self, env, action) -> None:
        raise NotImplementedError()

    def reset(self) -> None:
        pass
