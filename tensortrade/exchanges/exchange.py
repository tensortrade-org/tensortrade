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

import pandas as pd

from abc import abstractmethod
from typing import List

from tensortrade.base import Component, TimedIdentifiable


class Exchange(Component, TimedIdentifiable):
    """An abstract exchange for use within a trading environment."""
    registered_name = "exchanges"

    @property
    @abstractmethod
    def is_live(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_columns(self) -> List[str]:
        """The list of observation columns provided by the exchange each time step."""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_next_observation(self) -> bool:
        """If `False`, the exchange's data source has run out of observations.

        Resetting the exchange may be necessary to continue generating observations.

        Returns:
            Whether or not the specified instrument has a next observation.
        """
        raise NotImplementedError

    @abstractmethod
    def next_observation(self) -> pd.DataFrame:
        """Generate the next observation from the exchange, including wallet balances if specified.

        Returns:
            A `pandas.DataFrame` of exchange observations for the next time step.
        """
        raise NotImplementedError

    @abstractmethod
    def quote_price(self, trading_pair: 'TradingPair') -> float:
        """The quote price of a trading pair on the exchange, denoted in the base instrument.

        Arguments:
            trading_pair: The `TradingPair` to get the quote price for.

        Returns:
            The quote price of the specified trading pair, denoted in the base instrument.
        """
        raise NotImplementedError

    @abstractmethod
    def is_pair_tradeable(self, trading_pair: 'TradingPair') -> bool:
        """Whether or not the specified trading pair is tradeable on this exchange.

        Args:
            trading_pair: The `TradingPair` to test the tradeability of.

        Returns:
            A bool designating whether or not the pair is tradeable.
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
