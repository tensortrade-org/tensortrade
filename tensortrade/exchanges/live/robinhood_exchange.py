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

from typing import Dict, List
from gym.spaces import Space, Box
from ccxt import Exchange

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import Exchange


class RobinhoodExchange(Exchange):
    """An exchange for trading using the Robinhood API."""

    def __init__(self,  **kwargs):
        super().__init__(dtype=self.default('dtype', np.float16, kwargs),
                         feature_pipeline=self.default('feature_pipeline', None, kwargs))
        # TODO: Initialize the Robinhood client

        self._observation_type = self.default('observation_type', 'ohlcv', kwargs)
        self._observation_symbol = self.default(
            'observation_symbols', ['AAPL', 'MSFT', 'GOOG'], kwargs)
        self._timeframe = self.default('timeframe', '10m', kwargs)
        self._window_size = self.default('window_size', 1, kwargs)

        self._async_timeout_in_ms = self.default('async_timeout_in_ms', 15, kwargs)
        self._max_trade_wait_in_sec = self.default('max_trade_wait_in_sec', 60, kwargs)

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

    def _next_observation(self) -> pd.DataFrame:
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
