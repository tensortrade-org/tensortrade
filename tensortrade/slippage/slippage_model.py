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
# limitations under the License.

from abc import abstractmethod

from tensortrade import Component
from tensortrade.trades import Trade


class SlippageModel(Component):
    """A model for simulating slippage on an exchange trade."""

    registered_name = "slippage"

    def __init__(self):
        pass

    @abstractmethod
    def slip_up(self, number: float = 0, percentage: float = 0) -> float:
        slip = percentage/100
        return (number + (number*slip))

    @abstractmethod
    def slip_down(self, number: float = 0, percentage: float = 0) -> float:
        slip = percentage/100
        return (number - (number*slip))
