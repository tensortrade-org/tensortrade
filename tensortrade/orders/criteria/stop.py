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


from typing import Union
from enum import Enum

from tensortrade.orders.criteria import Criteria


class StopDirection(Enum):
    UP = 'up'
    DOWN = 'down'

    def __str__(self):
        return str(self.value)


class Stop(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is above or below a specific price."""

    def __init__(self, direction: Union[StopDirection, str], percent: float):
        self.direction = StopDirection(direction)
        self.percent = percent

    def call(self, order: 'Order', exchange: 'Exchange') -> bool:
        price = exchange.quote_price(order.pair)
        percent = abs(price - order.price) / price

        is_take_profit = (self.direction == StopDirection.UP) and (price >= order.price)
        is_stop_loss = (self.direction == StopDirection.DOWN) and (price <= order.price)

        return (is_take_profit or is_stop_loss) and percent >= self.percent

    def __str__(self):
        return '<Stop: direction={0}, percent={1}>'.format(self.direction,
                                                           self.percent)
