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


from tensortrade.orders.criteria import Criteria
from tensortrade.trades import TradeSide


class Limit(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price, hidden from the public order book."""

    def __init__(self, limit_price: float):
        self.limit_price = limit_price

    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        price = exchange.quote_price(order.pair)

        buy_satisfied = (order.side == TradeSide.BUY and price <= self.limit_price)
        sell_satisfied = (order.side == TradeSide.SELL and price >= self.limit_price)

        return buy_satisfied or sell_satisfied

    def __str__(self):
        return '<Limit: price={0}>'.format(self.limit_price)
