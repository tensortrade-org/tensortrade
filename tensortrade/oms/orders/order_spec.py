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


from typing import Callable

from tensortrade.core import Identifiable
from tensortrade.oms.orders import Order, TradeSide, TradeType


class OrderSpec(Identifiable):
    """A class for order creation following an order being complete.

    Parameters
    ----------
    side : `TradeSide`
        The trading side of the specification.
    trade_type : `TradeType`
        The type of trade for the specification.
    exchange_pair : `ExchangePair`
        The exchange pair for the specification.
    criteria : `Callable[[Order, Exchange], bool]`
        The criteria for executing the order after its been created.
    """

    def __init__(self,
                 side: 'TradeSide',
                 trade_type: 'TradeType',
                 exchange_pair: 'ExchangePair',
                 criteria: 'Callable[[Order, Exchange], bool]' = None):
        self.side = side
        self.type = trade_type
        self.exchange_pair = exchange_pair
        self.criteria = criteria

    def create_order(self, order: 'Order') -> 'Order':
        """Creates an order following from another order.

        Parameters
        ----------
        order : `Order`
            The previous order in the order path.

        Returns
        -------
        `Order`
            The order created from the specification parameters and the
            parameters of `order`.
        """

        wallet_instrument = self.side.instrument(self.exchange_pair.pair)

        exchange = order.exchange_pair.exchange
        wallet = order.portfolio.get_wallet(exchange.id, instrument=wallet_instrument)
        quantity = wallet.locked.get(order.path_id, None)

        if not quantity or quantity.size == 0:
            return None

        return Order(step=exchange.clock.step,
                     side=self.side,
                     trade_type=self.type,
                     exchange_pair=self.exchange_pair,
                     quantity=quantity,
                     portfolio=order.portfolio,
                     price=self.exchange_pair.price,
                     criteria=self.criteria,
                     end=order.end,
                     path_id=order.path_id)

    def to_dict(self) -> dict:
        """Creates dictionary representation of specification.

        Returns
        -------
        dict
            The dictionary representation of specification.
        """
        return {
            "id": self.id,
            "type": self.type,
            "exchange_pair": self.exchange_pair,
            "criteria": self.criteria
        }

    def __str__(self) -> str:
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self) -> str:
        return str(self)
