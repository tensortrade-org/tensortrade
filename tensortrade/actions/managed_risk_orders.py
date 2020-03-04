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


from typing import Union, List
from itertools import product
from gym.spaces import Discrete

from tensortrade.actions import ActionScheme
from tensortrade.orders import TradeSide, TradeType, Order, OrderListener, risk_managed_order


class ManagedRiskOrders(ActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.
    """

    def __init__(self,
                 stop_loss_percentages: Union[List[float], float] = [0.02, 0.04, 0.06],
                 take_profit_percentages: Union[List[float], float] = [0.01, 0.02, 0.03],
                 trade_sizes: Union[List[float], int] = 10,
                 durations: Union[List[int], int] = None,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):
        """
        Arguments:
            pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            stop_loss_percentages: A list of possible stop loss percentages for each order.
            take_profit_percentages: A list of possible take profit percentages for each order.
            trade_sizes: A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        super().__init__()
        self.stop_loss_percentages = self.default('stop_loss_percentages', stop_loss_percentages)
        self.take_profit_percentages = self.default(
            'take_profit_percentages', take_profit_percentages)
        self.trade_sizes = self.default('trade_sizes', trade_sizes)
        self.durations = self.default('durations', durations)
        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

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
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
        """
        return self._trade_sizes

    @trade_sizes.setter
    def trade_sizes(self, trade_sizes: Union[List[float], int]):
        self._trade_sizes = trade_sizes if isinstance(trade_sizes, list) else [
            (x + 1) / trade_sizes for x in range(trade_sizes)]

    @property
    def durations(self) -> List[int]:
        """A list of durations to select from when submitting an order."""
        return self._durations

    @durations.setter
    def durations(self, durations: Union[List[int], int]):
        self._durations = durations if isinstance(durations, list) else [durations]

    def compile(self):
        self.actions = list(product(self._stop_loss_percentages,
                                    self._take_profit_percentages,
                                    self._trade_sizes,
                                    self._durations,
                                    [TradeSide.BUY, TradeSide.SELL]))
        self.actions = list(product(self.exchange_pairs, self.actions))
        self.actions = [None] + self.actions

        self._action_space = Discrete(len(self.actions))

    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:

        if action == 0:
            return None

        (exchange_pair, (stop_loss, take_profit, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(exchange_pair.pair)
        wallet = portfolio.get_wallet(exchange_pair.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if size < 10 ** -exchange_pair.pair.base.precision:
            return None

        params = {
            'step': self.clock.step,
            'side': side,
            'exchange_pair': exchange_pair,
            'price': exchange_pair.price,
            'quantity': quantity,
            'down_percent': stop_loss,
            'up_percent': take_profit,
            'portfolio': portfolio,
            'trade_type': self._trade_type,
            'end': self.clock.step + duration if duration else None
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return order

    def reset(self):
        pass
