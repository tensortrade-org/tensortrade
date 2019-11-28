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
import uuid
import random

class Trade(object):
    """A trade object for use within trading environments."""

    def __init__(self, step: int, symbol: str, trade_type: 'TradeType', amount: float, price: float, **kwargs):
        """
        Arguments:
            step: The timestep the trade was made during the trading episode.
            symbol: The exchange symbol of the instrument in the trade (AAPL, ETH/USD, NQ1!, etc).
            trade_type: The type of trade executed (0 = HOLD, 1=LIMIT_BUY, 2=MARKET_BUY, 3=LIMIT_SELL, 4=MARKET_SELL).
            amount: The amount of the instrument in the trade (shares, satoshis, contracts, etc).
            price: The price paid per instrument in terms of the base instrument (e.g. 10000 represents $10,000.00 if the `base_instrument` is "USD").
        """
        self._step = step
        self._id = kwargs.get('id', uuid.uuid1(random.randrange(0,1e8)))
        self.symbol = symbol
        self._trade_type = trade_type

        self.order_price = price
        self.order_amount = amount

        self.order_commission_percent = kwargs.get('order_commission_percent', 0.0075)
        self.order_commission = kwargs.get('order_commission', 0)

        self.transact_price = kwargs.get('transact_price', 0)
        self.transact_amount = kwargs.get('trade_amount', 0)
        self.transact_commission_percent = kwargs.get('transact_commission_percent', 0.0075)
        self.transact_commission = kwargs.get('transact_commission', 0)

        self.order_total = kwargs.get('order_total', 0)
        self.transact_total = kwargs.get('transact_total', 0)
        self.valid = False
        self.executed = False

    @property
    def step(self) -> str:
        """The timestep the trade was made during the trading episode."""
        return self._step

    @step.setter
    def step(self, step: str):
        self._step = step

    def copy(self) -> 'Trade':
        """Return a copy of the current trade object."""
        return Trade(id=self._id,
                    step= self.step,
                    symbol=self.symbol,
                    trade_type=self._trade_type,
                    amount=self.amount,
                    price=self.price,
                    order_price=self.order_price,
                    order_amount=self.order_amount,
                    order_commission=self.order_commission,
                    order_commission_percent=self.order_commission_percent,
                    transact_price=self.transact_price,
                    transact_amount=self.transact_amount,
                    transact_commission=self.transact_commission,
                    transact_commission_percent=self.transact_commission_percent,
                    transact_total=self.transact_total,
                    order_total=self.order_total,
                    valid=self.valid,
                    executed=self.executed
                    )

    @property
    def trade_type(self):
        return self._trade_type

    @property
    def to_dict(self) -> dict:
        """Return a dict of the current trade object."""
        t = {'id':self._id,
            'step':self.step,
            'symbol':self.symbol,
            'trade_type':self._trade_type,
            'amount':self.amount,
            'price':self.price,
            'order_price':self.order_price,
            'order_amount':self.order_amount,
            'order_commission':self.order_commission,
            'order_commission_percent':self.order_commission_percent,
            'transact_price':self.transact_price,
            'transact_amount':self.transact_amount,
            'transact_commission':self.transact_commission,
            'transact_commission_percent':self.transact_commission_percent,
            'transact_total':self.transact_total,
            'order_total':self.order_total,
            'valid':self.valid,
            'executed':self.executed}
        return t

    @property
    def amount(self) -> float:
        PendingDeprecationWarning("Trade.Amount will be deprecated. Please use: Trade.order_amount | Trade.transact_amount")
        if self.transact_amount > 0:
            return self.transact_amount
        else:
            return self.order_amount

    @property
    def price(self) -> float:
        PendingDeprecationWarning("Trade.Amount will be deprecated. Please use: Trade.order_price | Trade.transact_price")
        if self.transact_price > 0:
            return self.transact_price
        else:
            return self.order_price

    @property
    def log(self) -> str:
        if self.is_buy:
            return self._buy_log()
        elif self.is_sell:
            return self._sell_log()
        else:
            return self._hold_log()

    def _buy_log(self) -> str:
        """Provides a log of the buy transaction"""
        log = "{}:{} {} {} units of {} for {} each. Total:{} Comission:{} @ {}%".format(
            self._id,
            self.step,
            self._trade_type,
            self.transact_amount,
            self.symbol,
            self.transact_price,
            self.transact_total,
            self.transact_commission,
            self.transact_commission_percent
        )
        return log

    def _sell_log(self) -> str:
        """Provides a log of the sell transaction"""
        log = "{}:{} {} {} units of {} for {} each. Total:{} Comission:{} @ {}%".format(
            self._id,
            self.step,
            self._trade_type,
            self.transact_amount,
            self.symbol,
            self.transact_price,
            self.transact_total,
            self.transact_commission,
            self.transact_commission_percent
        )
        return log

    def _hold_log(self) -> str:
        """Provides a log of the hold transaction"""
        log = "{}:{} {} {} units of {} at {} each. Total:{}".format(
            self._id,
            self.step,
            self._trade_type,
            self.transact_amount,
            self.symbol,
            self.transact_price,
            self.transact_total
        )
        return log

    @property
    def is_hold(self) -> bool:
        """
        Returns:
            Whether the trade type is non-existent (i.e. hold).
        """
        return self._trade_type.is_hold

    @property
    def is_buy(self) -> bool:
        """
        Returns:
            Whether the trade type is a buy offer.
        """
        return self._trade_type.is_buy

    @property
    def is_sell(self) -> bool:
        """
        Returns:
            Whether the trade type is a sell offer.
        """
        return self._trade_type.is_sell

    @property
    def is_limit_buy(self) -> bool:
        """
        Returns:
            Whether the trade type is a buy offer.
        """
        return self._trade_type.is_limit_buy

    @property
    def is_market_buy(self) -> bool:
        """
        Returns:
            Whether the trade type is a buy offer.
        """
        return self._trade_type.is_market_buy

    @property
    def is_limit_sell(self) -> bool:
        """
        Returns:
            Whether the trade type is a sell offer.
        """
        return self._trade_type.is_limit_sell

    @property
    def is_market_sell(self) -> bool:
        """
        Returns:
            Whether the trade type is a sell offer.
        """
        return self._trade_type.is_market_sell
