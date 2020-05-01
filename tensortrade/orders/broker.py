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

from typing import List, Dict
from collections import OrderedDict

from tensortrade.base.core import TimeIndexed
from .order import Order, OrderStatus
from .order_listener import OrderListener


class Broker(OrderListener, TimeIndexed):
    """A broker for handling the execution of orders on multiple exchanges.
    Orders are kept in a virtual order book until they are ready to be executed.
    """

    def __init__(self):
        self._unexecuted = []
        self._executed = {}
        self._trades = OrderedDict()

    @property
    def unexecuted(self) -> List[Order]:
        """The list of orders the broker is waiting to execute, when their criteria is satisfied."""
        return self._unexecuted

    @property
    def executed(self) -> Dict[str, Order]:
        """The dictionary of orders the broker has executed since resetting, organized by order id"""
        return self._executed

    @property
    def trades(self) -> Dict[str, 'Trade']:
        """The dictionary of trades the broker has executed since resetting, organized by order id."""
        return self._trades

    def submit(self, order: Order):
        self._unexecuted += [order]

    def cancel(self, order: Order):
        if order.status == OrderStatus.CANCELLED:
            raise Warning(
                'Cannot cancel order {} - order has already been cancelled.'.format(order.id))

        if order in self._unexecuted:
            self._unexecuted.remove(order)

        order.cancel()

    def update(self):
        for order in self._unexecuted:
            if order.is_executable():
                self._unexecuted.remove(order)
                self._executed[order.id] = order

                order.attach(self)
                order.execute()

        for order in self._unexecuted + list(self._executed.values()):
            if order.is_active() and order.is_expired():
                self.cancel(order)

    def on_fill(self, order: Order, trade: 'Trade'):
        if trade.order_id in self._executed and trade not in self._trades:
            self._trades[trade.order_id] = self._trades.get(trade.order_id, [])
            self._trades[trade.order_id] += [trade]

            if order.is_complete():
                next_order = order.complete()

                if next_order:
                    if next_order.is_executable():
                        self._executed[next_order.id] = next_order

                        next_order.attach(self)
                        next_order.execute()
                    else:
                        self.submit(next_order)

    def reset(self):
        self._unexecuted = []
        self._executed = {}
        self._trades = OrderedDict()
