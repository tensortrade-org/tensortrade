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


from typing import Callable
from decimal import Decimal

from tensortrade.core import Component, TimedIdentifiable
from tensortrade.oms.instruments import TradingPair


class ExchangeOptions:
    """An options class to specify the settings of an exchange.

    Parameters
    ----------
    commission : float, default 0.003
        The percentage of the order size taken by the exchange.
    min_trade_size : float, default 1e-6
        The minimum trade size an order can have.
    max_trade_size : float, default 1e6
        The maximum trade size an order can have.
    min_trade_price : float, default 1e-8
        The minimum price an exchange can have.
    max_trade_price : float, default 1e8
        The maximum price an exchange can have.
    is_live : bool, default False
        Whether live orders should be submitted to the exchange.
    """

    def __init__(self,
                 commission: float = 0.003,
                 min_trade_size: float = 1e-6,
                 max_trade_size: float = 1e6,
                 min_trade_price: float = 1e-8,
                 max_trade_price: float = 1e8,
                 is_live: bool = False):
        self.commission = commission
        self.min_trade_size = min_trade_size
        self.max_trade_size = max_trade_size
        self.min_trade_price = min_trade_price
        self.max_trade_price = max_trade_price
        self.is_live = is_live


class Exchange(Component, TimedIdentifiable):
    """An abstract exchange for use within a trading environment.

    Parameters
    ----------
    name : str
        The name of the exchange.
    service : `Union[Callable, str]`
        The service to be used for filling orders.
    options : `ExchangeOptions`
        The options used to specify the setting of the exchange.
    """

    registered_name = "exchanges"

    def __init__(self,
                 name: str,
                 service: Callable,
                 options: ExchangeOptions = None):
        super().__init__()
        self.name = name
        self._service = service
        self.options = options if options else ExchangeOptions()
        self._price_streams = {}

    def __call__(self, *streams) -> "Exchange":
        """Sets up the price streams used to generate the prices.

        Parameters
        ----------
        *streams
            The positional arguments each being a price stream.

        Returns
        -------
        `Exchange`
            The exchange the price streams were passed in for.
        """
        for s in streams:
            pair = "".join([c if c.isalnum() else "/" for c in s.name])
            self._price_streams[pair] = s.rename(self.name + ":/" + s.name)
        return self

    def streams(self) -> "List[Stream[float]]":
        """Gets the price streams for the exchange.

        Returns
        -------
        `List[Stream[float]]`
            The price streams for the exchange.
        """
        return list(self._price_streams.values())

    def quote_price(self, trading_pair: "TradingPair") -> "Decimal":
        """The quote price of a trading pair on the exchange, denoted in the
        core instrument.

        Parameters
        ----------
        trading_pair : `TradingPair`
            The trading pair to get the quote price for.

        Returns
        -------
        `Decimal`
            The quote price of the specified trading pair, denoted in the core instrument.
        """
        price = Decimal(self._price_streams[str(trading_pair)].value)
        if price == 0:
            raise ValueError("Price of trading pair {} is 0. Please check your input data to make sure there always is "
                             "a valid (nonzero) price.".format(trading_pair))

        price = price.quantize(Decimal(10) ** -trading_pair.base.precision)
        if price == 0:
            raise ValueError("Price quantized in base currency precision ({}) would amount to 0 {}. "
                             "Please consider defining a custom instrument with a higher precision."
                             .format(trading_pair.base.precision, trading_pair.base))

        return price

    def is_pair_tradable(self, trading_pair: 'TradingPair') -> bool:
        """Whether or not the specified trading pair is tradable on this
        exchange.

        Parameters
        ----------
        trading_pair : `TradingPair`
            The trading pair to test the tradability of.

        Returns
        -------
        bool
            Whether or not the pair is tradable.
        """
        return str(trading_pair) in self._price_streams.keys()

    def execute_order(self, order: 'Order', portfolio: 'Portfolio') -> None:
        """Execute an order on the exchange.

        Parameters
        ----------
        order: `Order`
            The order to execute.
        portfolio : `Portfolio`
            The portfolio to use.
        """
        trade = self._service(
            order=order,
            base_wallet=portfolio.get_wallet(self.id, order.pair.base),
            quote_wallet=portfolio.get_wallet(self.id, order.pair.quote),
            current_price=self.quote_price(order.pair),
            options=self.options,
            clock=self.clock
        )

        if trade:
            order.fill(trade)
