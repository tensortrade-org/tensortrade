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


import numpy as np
import pandas as pd
import tensortrade.slippage as slippage

from typing import List, Callable, Dict

from tensortrade.base.exceptions import InsufficientFundsForAllocation
from tensortrade.data import DataSource
from tensortrade.trades import Trade, TradeType, TradeSide
from tensortrade.instruments import TradingPair, Quantity, Price
from tensortrade.exchanges import Exchange
from tensortrade.instruments import USD, BTC


class SimulatedExchange(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environments.
    """

    def __init__(self, source: DataSource, extract: Callable[[dict], Dict[TradingPair, Price]], **kwargs):
        super().__init__(source, extract)
        self._commission = self.default('commission', 0.003, kwargs)
        self._base_instrument = self.default('base_instrument', USD, kwargs)
        self._quote_instrument = self.default('quote_instrument', BTC, kwargs)
        self._min_trade_size = self.default('min_trade_size', 1e-6, kwargs)
        self._max_trade_size = self.default('max_trade_size', 1e6, kwargs)
        self._min_trade_price = self.default('min_trade_price', 1e-8, kwargs)
        self._max_trade_price = self.default('max_trade_price', 1e8, kwargs)

        slippage_model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(slippage_model) if isinstance(
            slippage_model, str) else slippage_model()

    @property
    def is_live(self):
        return False

    def is_pair_tradable(self, pair: TradingPair) -> bool:
        return pair.base == self._base_instrument and pair.quote == self._quote_instrument

    def _contain_price(self, price: float) -> float:
        return max(min(price, self._max_trade_price), self._min_trade_price)

    def _contain_size(self, size: float) -> float:
        return max(min(size, self._max_trade_size), self._min_trade_size)

    def _execute_buy_order(self, order: 'Order', base_wallet: 'Wallet', quote_wallet: 'Wallet', current_price: float) -> Trade:
        price = self._contain_price(current_price)

        if order.type == TradeType.LIMIT and order.price < current_price:
            return None

        commission = Quantity(order.pair.base, order.size * self._commission, order.path_id)
        base_size = self._contain_size(order.size - commission.size)

        if order.type == TradeType.MARKET:
            scale = order.price / price
            base_size = self._contain_size(scale * order.size - commission.size)

        base_wallet -= commission

        try:
            quantity = Quantity(order.pair.base, base_size, order.path_id)
            base_wallet -= quantity
        except InsufficientFundsForAllocation:
            balance = base_wallet.locked[order.path_id]
            quantity = Quantity(order.pair.base, balance.size, order.path_id)
            base_wallet -= quantity

        quote_size = (order.price / price) * (quantity.size / price)
        quote_wallet += Quantity(order.pair.quote, quote_size, order.path_id)

        trade = Trade(order_id=order.id,
                      exchange_id=self.id,
                      step=self.clock.step,
                      pair=order.pair,
                      side=TradeSide.BUY,
                      trade_type=order.type,
                      quantity=quantity,
                      price=price,
                      commission=commission)

        # self._slippage_model.adjust_trade(trade)

        return trade

    def _execute_sell_order(self,
                            order: 'Order',
                            base_wallet: 'Wallet',
                            quote_wallet: 'Wallet',
                            current_price: 'Price') -> Trade:
        price = self._contain_price(current_price)

        if order.type == TradeType.LIMIT and order.price > current_price:
            return None

        commission = Quantity(order.pair.base, order.size * self._commission, order.path_id)
        size = self._contain_size(order.size - commission.size)
        quantity = Quantity(order.pair.base, size, order.path_id)

        try:
            quote_size = quantity.size / price * (price / order.price)
            quote_wallet -= Quantity(order.pair.quote, quote_size, order.path_id)
        except InsufficientFundsForAllocation:
            balance = quote_wallet.locked[order.path_id]
            quantity = Quantity(order.pair.quote, balance.size, order.path_id)
            quote_wallet -= quantity

        base_wallet += quantity
        base_wallet -= commission

        trade = Trade(order_id=order.id,
                      exchange_id=self.id,
                      step=self.clock.step,
                      pair=order.pair,
                      side=TradeSide.SELL,
                      trade_type=order.type,
                      quantity=quantity,
                      price=price,
                      commission=commission)

        # self._slippage_model.adjust_trade(trade)

        return trade

    def execute_order(self, order: 'Order', portfolio: 'Portfolio'):
        base_wallet = portfolio.get_wallet(self.id, order.pair.base)
        quote_wallet = portfolio.get_wallet(self.id, order.pair.quote)
        current_price = self.quote_price(order.pair)
        print(current_price)

        if order.is_buy:
            trade = self._execute_buy_order(order, base_wallet, quote_wallet, current_price)
        elif order.is_sell:
            trade = self._execute_sell_order(order, base_wallet, quote_wallet, current_price)
        else:
            trade = None

        if trade:
            order.fill(self, trade)

    def reset(self):
        pass
