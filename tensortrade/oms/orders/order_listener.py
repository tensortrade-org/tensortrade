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


from abc import ABCMeta


class OrderListener(object, metaclass=ABCMeta):
    """A callback class for an order."""

    def on_execute(self, order: 'Order') -> None:
        """Callback for an order after execution.

        Parameters
        ----------
        order : `Order`
            The executed order.
        """
        pass

    def on_cancel(self, order: 'Order') -> None:
        """Callback for an order after cancellation.

        Parameters
        ----------
        order : `Order`
            The cancelled order.
        """
        pass

    def on_fill(self, order: 'Order', trade: 'Trade') -> None:
        """Callback for an order after being filled.

        Parameters
        ----------
        order : `Order`
            The order being filled.
        trade : `Trade`
            The trade that is filling the order.
        """
        pass

    def on_complete(self, order: 'Order') -> None:
        """Callback for an order after being completed.

        Parameters
        ----------
        order : `Order`
            The completed order.
        """
        pass
