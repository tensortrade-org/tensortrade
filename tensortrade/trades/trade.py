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

from enum import Enum


class TradeSide(Enum):
    BUY = 'buy'
    SELL = 'sell'


class Trade(object):
    """A trade object for use within trading environments."""

    def __init__(self, order_id: str, exchange_id: str, step: int, pair: 'TradingPair', side: TradeSide, amount: float, price: float):
        """
        Arguments:
            order_id: The id of the order that created the trade.
            order_id: The id of the exchange the trade was executed on.
            step: The timestep the trade was made during the trading episode.
            pair: The trading pair of the instruments in the trade.
            (e.g. BTC/USDT, ETH/BTC, ADA/BTC, AAPL/USD, NQ1!/USD, CAD/USD, etc)
            side: Whether the quote instrument is being bought or sold.
            (e.g. BUY = trade the `base_instrument` for the `quote_instrument` in the pair. SELL = trade the `quote_instrument` for the `base_instrument`)
            amount: The amount of the base instrument in the trade.
            (e.g. 1000 shares, 6.50 satoshis, 2.3 contracts, etc).
            price: The price paid per quote instrument in terms of the base instrument.
            (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
        """
        self.order_id = order_id
        self.exchange_id = exchange_id
        self.step = step
        self.pair = pair
        self.side = side
        self.amount = amount
        self.price = price

    def copy(self) -> 'Trade':
        """Return a copy of the current trade object."""
        return Trade(self.order_id, self.exchange_id, self.step, self.pair, self.side, self.amount, self.price)

    @property
    def is_buy(self) -> bool:
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == TradeSide.SELL
