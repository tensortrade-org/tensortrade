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

from tensortrade.base import TimedIdentifiable


class TradeType(Enum):
    LIMIT = 'limit'
    MARKET = 'market'

    def __str__(self):
        return str(self.value)


class TradeSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

    def __str__(self):
        return str(self.value)


class Trade(TimedIdentifiable):
    """A trade object for use within trading environments."""

    def __init__(self,
                 order_id: str,
                 exchange_id: str,
                 step: int,
                 pair: 'TradingPair',
                 side: TradeSide,
                 trade_type: TradeType,
                 quantity: 'Quantity',
                 price: float,
                 commission: float):
        """
        Arguments:
            order_id: The id of the order that created the trade.
            exchange_id: The id of the exchange the trade was executed on.
            step: The timestep the trade was made during the trading episode.
            pair: The trading pair of the instruments in the trade.
            (e.g. BTC/USDT, ETH/BTC, ADA/BTC, AAPL/USD, NQ1!/USD, CAD/USD, etc)
            side: Whether the quote instrument is being bought or sold.
            (e.g. BUY = trade the `base_instrument` for the `quote_instrument` in the pair. SELL = trade the `quote_instrument` for the `base_instrument`)
            size: The size of the base instrument in the trade.
            (e.g. 1000 shares, 6.50 satoshis, 2.3 contracts, etc).
            price: The price paid per quote instrument in terms of the base instrument.
            (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
            commission: The commission paid for the trade in terms of the base instrument.
            (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
        """
        self.order_id = order_id
        self.exchange_id = exchange_id
        self.step = step
        self.pair = pair
        self.side = side
        self.type = trade_type
        self.quantity = quantity
        self.price = price
        self.commission = commission

    @property
    def base_instrument(self) -> 'Instrument':
        return self.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.pair.quote

    @property
    def size(self) -> float:
        return self.quantity.size

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, price: float):
        self._price = round(price, self.pair.base.precision)

    @property
    def commission(self) -> 'Quantity':
        return self._commission

    @commission.setter
    def commission(self, commission: float):
        self._commission = commission.size * self.pair.base

    @property
    def is_buy(self) -> bool:
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == TradeSide.SELL

    @property
    def is_limit_order(self) -> bool:
        return self.type == TradeType.LIMIT

    @property
    def is_market_order(self) -> bool:
        return self.type == TradeType.MARKET

    def to_dict(self):
        return {'id': self.id,
                'order_id': self.order_id,
                'step': self.step,
                'base_symbol': self.pair.base.symbol,
                'quote_symbol': self.pair.quote.symbol,
                'side': self.side,
                'type': self.type,
                'quantity': self.quantity,
                'price': self.price,
                'commission': self.commission
                }

    def to_json(self):
        return {'id': str(self.id),
                'order_id': str(self.order_id),
                'step': str(self.step),
                'base_symbol': str(self.pair.base.symbol),
                'quote_symbol': str(self.pair.quote.symbol),
                'side': str(self.side),
                'type': str(self.type),
                'quantity': str(self.quantity),
                'price': str(self.price),
                'commission': str(self.commission)
                }

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
