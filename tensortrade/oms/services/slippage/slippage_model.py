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

from tensortrade.core import Component
from tensortrade.oms.orders import Trade


class SlippageModel(Component):
    """A model for simulating slippage on an exchange trade."""

    registered_name = "slippage"

    def __init__(self):
        pass

    @abstractmethod
    def adjust_trade(self, trade: Trade, **kwargs) -> Trade:
        """Simulate slippage on a trade ordered on a specific exchange.

        Parameters
        ----------
        trade : `Trade`
            The trade executed on the exchange.
        **kwargs : keyword arguments
            Any other arguments necessary for the model.

        Returns
        -------
        `Trade`
            A filled trade with the `price` and `size` adjusted for slippage.
        """
        raise NotImplementedError()
