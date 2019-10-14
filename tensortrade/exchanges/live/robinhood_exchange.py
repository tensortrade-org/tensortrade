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

import time
import numpy as np
import pandas as pd

from typing import Dict, List, Generator
from gym.spaces import Space, Box
from ccxt import Exchange

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import InstrumentExchange


class RobinhoodExchange(InstrumentExchange):
    """An instrument exchange for trading using the Robinhood API."""

    def __init__(self,  **kwargs):
        super().__init__(base_instrument=kwargs.get('base_instrument', 'USD'),
                         dtype=kwargs.get('dtype', np.float16),
                         feature_pipeline=kwargs.get('feature_pipeline', None))
        # TODO: Initialize the Robinhood client

        self._observation_type = kwargs.get('observation_type', 'ohlcv')
        self._observation_symbol = kwargs.get('observation_symbols', ['AAPL', 'MSFT', 'GOOG'])
        self._timeframe = kwargs.get('timeframe', '10m')
        self._window_size = kwargs.get('window_size', 1)

        self._async_timeout_in_ms = kwargs.get('async_timeout_in_ms', 15)
        self._max_trade_wait_in_sec = kwargs.get('max_trade_wait_in_sec', 60)

    @property
    def base_precision(self) -> float:
        # TODO
        raise NotImplementedError

    @base_precision.setter
    def base_precision(self, base_precision: float):
        raise ValueError('Cannot set the precision of `robinhood` exchanges.')

    @property
    def instrument_precision(self) -> float:
        # TODO
        raise NotImplementedError

    @instrument_precision.setter
    def instrument_precision(self, instrument_precision: float):
        raise ValueError('Cannot set the precision of `robinhood` exchanges.')

    @property
    def initial_balance(self) -> float:
        # TODO
        raise NotImplementedError

    @property
    def balance(self) -> float:
        # TODO
        raise NotImplementedError

    @property
    def portfolio(self) -> Dict[str, float]:
        # TODO
        raise NotImplementedError

    @property
    def trades(self) -> List[Trade]:
        # TODO
        raise NotImplementedError

    @property
    def performance(self) -> pd.DataFrame:
        # TODO
        raise NotImplementedError

    @property
    def generated_space(self) -> Space:
        # TODO
        raise NotImplementedError

    @property
    def has_next_observation(self) -> bool:
        # TODO
        raise NotImplementedError

    @property
    def _observation_generator(self) -> Generator[pd.DataFrame, None, None]:
        # TODO
        raise NotImplementedError

    def current_price(self, symbol: str) -> float:
        # TODO
        raise NotImplementedError

    def execute_trade(self, trade: Trade) -> Trade:
        # TODO
        raise NotImplementedError

    def reset(self):
        # TODO
        raise NotImplementedError
