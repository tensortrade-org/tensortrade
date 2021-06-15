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

from tensortrade.core.base import TimeIndexed
from tensortrade.oms.orders.order import Order, OrderStatus
from tensortrade.oms.orders.order_listener import OrderListener


class Broker(OrderListener, TimeIndexed):
    """A broker for handling the execution of orders on multiple exchanges.
    Orders are kept in a virtual order book until they are ready to be executed.

    Attributes
    ----------
    unexecuted : `List[Order]`
        The list of orders the broker is waiting to execute, when their
        criteria is satisfied.
    executed : `Dict[str, Order]`
        The dictionary of orders the broker has executed since resetting,
        organized by order id.
    trades : `Dict[str, Trade]`
        The dictionary of trades the broker has executed since resetting,
        organized by order id.
    """

    def __init__(self):
        self.unexecuted = []
        self.executed = {}
        self.trades = OrderedDict()

    def submit(self, order: "Order") -> None:
        """Submits an order to the broker.

        Adds `order` to the queue of orders waiting to be executed.

        Parameters
        ----------
        order : `Order`
            The order to be submitted.
        """
        self.unexecuted += [order]

    def cancel(self, order: "Order") -> None:
        """Cancels an order.

        Parameters
        ----------
        order : `Order`
            The order to be canceled.
        """
        if order.status == OrderStatus.CANCELLED:
            raise Warning(f"Order {order.id} has already been cancelled.")

        if order in self.unexecuted:
            self.unexecuted.remove(order)

        order.cancel()

    def update(self) -> None:
        """Updates the brokers order management system.

        The broker will look through the unexecuted orders and if an order
        is ready to be executed the broker will submit it to the executed
        list and execute the order.

        Then the broker will find any orders that are active, but expired, and
        proceed to cancel them.
        """
        executed_ids = []
        for order in self.unexecuted:
            if order.is_executable:
                executed_ids.append(order.id)
                self.executed[order.id] = order

                order.attach(self)
                order.execute()
        
        for order_id in executed_ids:
            self.unexecuted.remove(self.executed[order_id])

        for order in self.unexecuted + list(self.executed.values()):
            if order.is_active and order.is_expired:
                self.cancel(order)

    def on_fill(self, order: "Order", trade: "Trade") -> None:
        """Updates the broker after an order has been filled.

        Parameters
        ----------
        order : `Order`
            The order that is being filled.
        trade : `Trade`
            The trade that is being made to fill the order.
        """
        if trade.order_id in self.executed and trade not in self.trades:
            self.trades[trade.order_id] = self.trades.get(trade.order_id, [])
            self.trades[trade.order_id] += [trade]

            if order.is_complete:
                next_order = order.complete()

                if next_order:
                    if next_order.is_executable:
                        self.executed[next_order.id] = next_order

                        next_order.attach(self)
                        next_order.execute()
                    else:
                        self.submit(next_order)

    def reset(self) -> None:
        """Resets the broker."""
        self.unexecuted = []
        self.executed = {}
        self.trades = OrderedDict()
