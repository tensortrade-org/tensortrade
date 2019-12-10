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

from typing import Union, List
from abc import abstractmethod
from itertools import product
from gym.spaces import Discrete

from tensortrade.actions import ActionScheme
from tensortrade.trades import TradeSide, TradeType
from tensortrade.instruments import Quantity
from tensortrade.orders.criteria import StopLossCriteria
from tensortrade.orders import Order, OrderListener


class ManagedRiskOrders(ActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.
    """

    def __init__(self,
                 pairs: Union[List['TradingPair'], 'TradingPair'],
                 stop_loss_percentages: Union[List[float], float] = [0.05, 0.10],
                 take_profit_percentages: Union[List[float], float] = [0.025, 0.05, 0.10],
                 trade_sizes: Union[List[float], int] = 10,
                 trade_side: TradeSide = TradeSide.BUY,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):
        """
        Arguments:
            pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            criteria: A list of order criteria to select from when submitting an order.
            (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
            trade_sizes: A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        self.pairs = self.default('pairs', pairs)
        self.stop_loss_percentages = self.default('stop_loss_percentages', stop_loss_percentages)
        self.take_profit_percentages = self.default(
            'take_profit_percentages', take_profit_percentages)
        self.trade_sizes = self.default('trade_sizes', trade_sizes)
        self._trade_side = self.default('trade_side', trade_side)
        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self.reset()

    @property
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        return Discrete(len(self._actions))

    @property
    def pairs(self) -> List['TradingPair']:
        """A list of trading pairs to select from when submitting an order.
        (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
        """
        return self._pairs

    @pairs.setter
    def pairs(self, pairs: Union[List['TradingPair'], 'TradingPair']):
        self._pairs = pairs if isinstance(pairs, list) else [pairs]

    @property
    def stop_loss_percentages(self) -> List[float]:
        """A list of order percentage losses to select a stop loss from when submitting an order.
        (e.g. 0.01 = sell if price drops 1%, 0.15 = 15%, etc.)
        """
        return self._stop_loss_percentages

    @stop_loss_percentages.setter
    def stop_loss_percentages(self, stop_loss_percentages: Union[List[float], float]):
        self._stop_loss_percentages = stop_loss_percentages if isinstance(
            stop_loss_percentages, list) else [stop_loss_percentages]

    @property
    def take_profit_percentages(self) -> List[float]:
        """A list of order percentage gains to select a take profit from when submitting an order.
        (e.g. 0.01 = sell if price rises 1%, 0.15 = 15%, etc.)
        """
        return self._take_profit_percentages

    @take_profit_percentages.setter
    def take_profit_percentages(self, take_profit_percentages: Union[List[float], float]):
        self._take_profit_percentages = take_profit_percentages if isinstance(
            take_profit_percentages, list) else [take_profit_percentages]

    @property
    def trade_sizes(self) -> List[float]:
        """A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
        """
        return self._trade_sizes

    @trade_sizes.setter
    def trade_sizes(self, trade_sizes: Union[List[float], int]):
        self._trade_sizes = trade_sizes if isinstance(trade_sizes, list) else [
            (x + 1) / trade_sizes for x in range(trade_sizes)]

    def get_order(self, action: int, exchange: 'Exchange', portfolio: 'Portfolio') -> Order:
        if action == 0:
            return None

        (pair, stop_loss, take_profit, size) = self._actions[action]

        instrument = pair.base if self._trade_side == TradeSide.BUY else pair.quote
        wallet = portfolio.get_wallet(exchange.id, instrument=instrument)
        price = exchange.quote_price(instrument)
        amount = min(wallet.balance.amount, (wallet.balance.amount * size))

        if amount < 10 ** -instrument.precision:
            return None

        quantity = amount * instrument

        wallet -= quantity

        risk_criteria = StopLossCriteria(direction='either',
                                         up_percent=take_profit,
                                         down_percent=stop_loss)

        risk_management_order = Order(side=TradeSide.SELL if self._trade_side == TradeSide.BUY else TradeSide.BUY,
                                      trade_type=TradeType.MARKET,
                                      pair=pair,
                                      price=price,
                                      quantity=quantity,
                                      portfolio=portfolio,
                                      criteria=risk_criteria)

        order = Order(side=self._trade_side,
                      trade_type=self._trade_type,
                      pair=pair,
                      price=price,
                      quantity=quantity,
                      portfolio=portfolio,
                      followed_by=risk_management_order)

        quantity.lock_for(order.id)

        wallet += quantity

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return order

    def reset(self):
        self._actions = [None]

        for trading_pair, stop_loss, take_profit, size in product(self._pairs,
                                                                  self._stop_loss_percentages,
                                                                  self._take_profit_percentages,
                                                                  self._trade_sizes):
            self._actions += [(trading_pair, stop_loss, take_profit, size)]
