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

from abc import abstractmethod
from typing import Dict, Callable

from tensortrade.base import Component, TimedIdentifiable
from tensortrade.data import DataSource
from tensortrade.instruments import TradingPair, Price


class Exchange(Component, TimedIdentifiable):
    """An abstract exchange for use within a trading environment."""

    registered_name = "exchanges"

    def __init__(self,
                 source: DataSource,
                 extract: Callable[[dict], Dict[TradingPair, Price]]):
        source.attach(self)
        self._extract = extract
        self._prices = None

    def on_next(self, data: dict):
        self._prices = self._extract(data)

    @property
    @abstractmethod
    def is_live(self):
        raise NotImplementedError()

    def quote_price(self, trading_pair: 'TradingPair') -> 'Price':
        """The quote price of a trading pair on the exchange, denoted in the base instrument.

        Arguments:
            trading_pair: The `TradingPair` to get the quote price for.

        Returns:
            The quote price of the specified trading pair, denoted in the base instrument.
        """
        if self._prices:
            return Price(self._prices[trading_pair], trading_pair)

    @abstractmethod
    def is_pair_tradable(self, trading_pair: 'TradingPair') -> bool:
        """Whether or not the specified trading pair is tradable on this exchange.

        Args:
            trading_pair: The `TradingPair` to test the tradability of.

        Returns:
            A bool designating whether or not the pair is tradable.
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_order(self, order: 'Order', portfolio: 'Portfolio'):
        """Execute an order on the exchange.

        Arguments:
            order: The order to execute.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        pass
