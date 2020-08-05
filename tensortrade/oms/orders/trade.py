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

from tensortrade.core import TimedIdentifiable


class TradeType(Enum):

    LIMIT: str = "limit"
    MARKET: str = "market"

    def __str__(self):
        return str(self.value)


class TradeSide(Enum):

    BUY: str = "buy"
    SELL: str = "sell"

    def instrument(self, pair: "TradingPair") -> "Instrument":
        return pair.base if self == TradeSide.BUY else pair.quote

    def __str__(self):
        return str(self.value)


class Trade(TimedIdentifiable):
    """A trade object for use within trading environments."""

    def __init__(self,
                 order_id: str,
                 step: int,
                 exchange_pair: 'ExchangePair',
                 side: TradeSide,
                 trade_type: TradeType,
                 quantity: 'Quantity',
                 price: float,
                 commission: 'Quantity'):
        """
        Arguments:
            order_id: The id of the order that created the trade.
            step: The timestep the trade was made during the trading episode.
            exchange_pair: The exchange pair of instruments in the trade.
            (e.g. BTC/USDT, ETH/BTC, ADA/BTC, AAPL/USD, NQ1!/USD, CAD/USD, etc)
            side: Whether the quote instrument is being bought or sold.
            (e.g. BUY = trade the `base_instrument` for the `quote_instrument` in the pair. SELL = trade the `quote_instrument` for the `base_instrument`)
            size: The size of the core instrument in the trade.
            (e.g. 1000 shares, 6.50 satoshis, 2.3 contracts, etc).
            price: The price paid per quote instrument in terms of the core instrument.
            (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
            commission: The commission paid for the trade in terms of the core instrument.
            (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
        """
        super().__init__()
        self.order_id = order_id
        self.step = step
        self.exchange_pair = exchange_pair
        self.side = side
        self.type = trade_type
        self.quantity = quantity
        self.price = price
        self.commission = commission

    @property
    def base_instrument(self) -> 'Instrument':
        return self.exchange_pair.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.exchange_pair.pair.quote

    @property
    def size(self) -> float:
        return self.quantity.size

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, price: float):
        self._price = price

    @property
    def commission(self) -> 'Quantity':
        return self._commission

    @commission.setter
    def commission(self, commission: 'Quantity'):
        self._commission = commission

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
                'exchange_pair': self.exchange_pair,
                'base_symbol': self.exchange_pair.pair.base.symbol,
                'quote_symbol': self.exchange_pair.pair.quote.symbol,
                'side': self.side,
                'type': self.type,
                'size': self.size,
                'quantity': self.quantity,
                'price': self.price,
                'commission': self.commission,
                "created_at": self.created_at
                }

    def to_json(self):
        return {'id': str(self.id),
                'order_id': str(self.order_id),
                'step': int(self.step),
                'exchange_pair': str(self.exchange_pair),
                'base_symbol': str(self.exchange_pair.pair.base.symbol),
                'quote_symbol': str(self.exchange_pair.pair.quote.symbol),
                'side': str(self.side),
                'type': str(self.type),
                'size': float(self.size),
                'quantity': str(self.quantity),
                'price': float(self.price),
                'commission': str(self.commission),
                "created_at": str(self.created_at)
                }

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
