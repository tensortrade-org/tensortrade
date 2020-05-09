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

import re
import pandas as pd

from typing import Callable, Tuple, Union, List

from tensortrade import Component, TimedIdentifiable
from tensortrade.instruments import Instrument, Quantity, ExchangePair
from tensortrade.data.stream.listeners import FeedListener

from .wallet import Wallet


WalletType = Union['Wallet', Tuple['Exchange', Instrument, float]]


class Portfolio(Component, TimedIdentifiable, FeedListener):
    """A portfolio of wallets on exchanges."""

    registered_name = "portfolio"

    def __init__(self,
                 base_instrument: Instrument,
                 wallets: List[WalletType] = None,
                 order_listener: 'OrderListener' = None,
                 performance_listener: Callable[[pd.DataFrame], None] = None):
        super().__init__()

        wallets = wallets or []

        self._base_instrument = self.default('base_instrument', base_instrument)
        self._order_listener = self.default('order_listener', order_listener)
        self._performance_listener = self.default('performance_listener', performance_listener)
        self._wallets = {}

        for wallet in wallets:
            self.add(wallet)

        self._initial_balance = self.base_balance
        self._initial_net_worth = None
        self._net_worth = None
        self._performance = None
        self._keys = None

    @property
    def base_instrument(self) -> Instrument:
        """The exchange instrument used to measure value and performance statistics."""
        return self._base_instrument

    @base_instrument.setter
    def base_instrument(self, base_instrument: Instrument):
        self._base_instrument = base_instrument

    @property
    def order_listener(self) -> Instrument:
        """The order listener to set for all orders executed by this portfolio."""
        return self._order_listener

    @order_listener.setter
    def order_listener(self, order_listener: 'OrderListener'):
        self._order_listener = order_listener

    @property
    def performance_listener(self) -> Instrument:
        """The performance listener to send all portfolio updates to."""
        return self._performance_listener

    @performance_listener.setter
    def performance_listener(self, performance_listener: Callable[[pd.DataFrame], None]):
        self._performance_listener = performance_listener

    @property
    def wallets(self) -> List[Wallet]:
        return list(self._wallets.values())

    @property
    def exchanges(self) -> List['Exchange']:
        exchanges = []
        for w in self.wallets:
            if w.exchange not in exchanges:
                exchanges += [w.exchange]
        return exchanges

    @property
    def ledger(self) -> 'Ledger':
        return Wallet.ledger

    @property
    def exchange_pairs(self) -> List['ExchangePair']:
        exchange_pairs = []
        for w in self.wallets:
            if w.instrument != self.base_instrument:
                exchange_pairs += [ExchangePair(w.exchange, self.base_instrument/w.instrument)]
        return exchange_pairs

    @property
    def initial_balance(self) -> Quantity:
        """The initial balance of the base instrument over all wallets, set by calling `reset`."""
        return self._initial_balance

    @property
    def base_balance(self) -> Quantity:
        """The current balance of the base instrument over all wallets."""
        return self.balance(self._base_instrument)

    @property
    def initial_net_worth(self):
        return self._initial_net_worth

    @property
    def net_worth(self) -> float:
        """Calculate the net worth of the active account on thenge.

        Returns:
            The total portfolio value of the active account on the exchange.
        """
        return self._net_worth

    @property
    def profit_loss(self) -> float:
        """Calculate the percentage change in net worth since the last reset.

        Returns:
            The percentage change in net worth since the last reset.
            i.e. A return value of 2 would indicate a 100% increase in net worth (e.g. $100 -> $200)
        """
        return self.net_worth / self.initial_net_worth

    @property
    def performance(self) -> pd.DataFrame:
        """The performance of the active account on the exchange since the last reset.

        Returns:
            A `pandas.DataFrame` with the locked and unlocked balance of each wallet at each time step.
        """
        return self._performance

    def balance(self, instrument: Instrument) -> Quantity:
        """The total balance of the portfolio in a specific instrument available for use."""
        balance = Quantity(instrument, 0)

        for (_, symbol), wallet in self._wallets.items():
            if symbol == instrument.symbol:
                balance += wallet.balance

        return balance

    def locked_balance(self, instrument: Instrument) -> Quantity:
        """The total balance of the portfolio in a specific instrument locked in orders."""
        balance = Quantity(instrument, 0)

        for (_, symbol), wallet in self._wallets.items():
            if symbol == instrument.symbol:
                balance += wallet.locked_balance

        return balance

    def total_balance(self, instrument: Instrument) -> Quantity:
        """The total balance of the portfolio in a specific instrument, both available for use and locked in orders."""
        return self.balance(instrument) + self.locked_balance(instrument)

    @property
    def balances(self) -> List[Quantity]:
        """The current unlocked balance of each instrument over all wallets."""
        return [wallet.balance for wallet in self._wallets.values()]

    @property
    def locked_balances(self) -> List[Quantity]:
        """The current locked balance of each instrument over all wallets."""
        return [wallet.locked_balance for wallet in self._wallets.values()]

    @property
    def total_balances(self) -> List[Quantity]:
        """The current total balance of each instrument over all wallets."""
        return [wallet.total_balance for wallet in self._wallets.values()]

    def get_wallet(self, exchange_id: str, instrument: Instrument):
        return self._wallets[(exchange_id, instrument.symbol)]

    def add(self, wallet: WalletType):
        if isinstance(wallet, tuple):
            wallet = Wallet.from_tuple(wallet)
        self._wallets[(wallet.exchange.id, wallet.instrument.symbol)] = wallet

    def remove(self, wallet: 'Wallet'):
        self._wallets.pop((wallet.exchange.id, wallet.instrument.symbol), None)

    def remove_pair(self, exchange: 'Exchange', instrument: Instrument):
        self._wallets.pop((exchange.id, instrument.symbol), None)

    @staticmethod
    def find_keys(data: dict):
        price_pattern = re.compile("\\w+:/([A-Z]{3,4}).([A-Z]{3,4})")
        endings = [
            ":/free",
            ":/locked",
            ":/total",
            "worth"
        ]

        keys = []
        for k in data.keys():
            if any(k.endswith(end) for end in endings):
                keys += [k]
            elif price_pattern.match(k):
                keys += [k]

        return keys

    def on_next(self, data: dict):
        if not self._keys:
            self._keys = self.find_keys(data)

        index = pd.Index([self.clock.step], name="step")
        performance_data = {k: data[k] for k in self._keys}
        performance_data['base_symbol'] = self.base_instrument.symbol
        performance_step = pd.DataFrame(performance_data, index=index)

        net_worth = data['net_worth']

        if self._performance is None:
            self._performance = performance_step
            self._initial_net_worth = net_worth
            self._net_worth = net_worth
        else:
            self._performance = self._performance.append(performance_step)
            self._net_worth = net_worth

        if self._performance_listener:
            self._performance_listener(performance_step)

    def reset(self):
        self._initial_balance = self.base_balance
        self._initial_net_worth = None
        self._net_worth = None
        self._performance = None

        self.ledger.reset()
        for wallet in self._wallets.values():
            wallet.reset()
