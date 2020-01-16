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


import tensortrade.slippage as slippage

import tensortrade.exchanges.services.execution.simulated as services

from tensortrade.data import ExchangeDataSource
from tensortrade.trades import TradeOptions
from tensortrade.exchanges import Exchange
from tensortrade.instruments import USD, BTC


class SimulatedExchange(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environments.
    """

    def __init__(self, source: ExchangeDataSource, **kwargs):
        super().__init__(source)
        self._commission = self.default('commission', 0.003, kwargs)

        self._trade_options = TradeOptions(
            min_trade_size=self.default('min_trade_size', 1e-6, kwargs),
            max_trade_size=self.default('max_trade_size', 1e6, kwargs),
            min_trade_price=self.default('min_trade_price', 1e-8, kwargs),
            max_trade_price=self.default('max_trade_price', 1e8, kwargs)
        )

        slippage_model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(slippage_model) if isinstance(
            slippage_model, str) else slippage_model()

    @property
    def is_live(self):
        return False

    def is_pair_tradable(self, pair: 'TradingPair') -> bool:
        return pair in self._ds.prices.keys()

    def execute_order(self, order: 'Order', portfolio: 'Portfolio'):

        trade = services.execute_order(
            order=order,
            base_wallet=portfolio.get_wallet(self.id, order.pair.base),
            quote_wallet=portfolio.get_wallet(self.id, order.pair.quote),
            current_price=self.quote_price(order.pair).rate,
            commission=self._commission,
            options=self._trade_options,
            exchange_id=self.id,
            clock=self.clock
        )

        if trade:
            order.fill(self, trade)

    def reset(self):
        pass
