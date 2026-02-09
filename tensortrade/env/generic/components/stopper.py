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

from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed


class Stopper(Component, TimeIndexed):
    """A component for determining if the environment satisfies a defined
    stopping criteria.
    """

    registered_name = "stopper"

    @abstractmethod
    def stop(self, env: 'TradingEnv') -> bool:
        """Computes if the environment satisfies the defined stopping criteria.

        Parameters
        ----------
        env : `TradingEnv`
            The trading environment.

        Returns
        -------
        bool
            If the environment should stop or continue.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the stopper."""
        pass
