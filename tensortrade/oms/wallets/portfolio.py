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

from typing import Callable, Tuple, List, TypeVar


from tensortrade.core import Component, TimedIdentifiable
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.orders import OrderListener
from tensortrade.oms.instruments import Instrument, Quantity, ExchangePair
from tensortrade.oms.wallets.wallet import Wallet
from tensortrade.oms.wallets.ledger import Ledger
from collections import OrderedDict


WalletType = TypeVar("WalletType", Wallet, Tuple[Exchange, Instrument, float])


class Portfolio(Component, TimedIdentifiable):
    """A portfolio of wallets on exchanges.

    Parameters
    ----------
    base_instrument : `Instrument`
        The exchange instrument used to measure value and performance statistics.
    wallets : `List[WalletType]`
        The wallets to be used in the portfolio.
    order_listener : `OrderListener`
        The order listener to set for all orders executed by this portfolio.
    performance_listener : `Callable[[OrderedDict], None]`
        The performance listener to send all portfolio updates to.
    """

    registered_name = "portfolio"

    def __init__(self,
                 base_instrument: Instrument,
                 wallets: List[WalletType] = None,
                 order_listener: 'OrderListener' = None,
                 performance_listener: Callable[[OrderedDict], None] = None):
        super().__init__()

        wallets = wallets or []

        self.base_instrument = self.default('base_instrument', base_instrument)
        self.order_listener = self.default('order_listener', order_listener)
        self.performance_listener = self.default('performance_listener', performance_listener)
        self._wallets = {}

        for wallet in wallets:
            self.add(wallet)

        self._initial_balance = self.base_balance
        self._initial_net_worth = None
        self._net_worth = None
        self._performance = None
        self._keys = None

    @property
    def wallets(self) -> 'List[Wallet]':
        """All the wallets in the portfolio. (`List[Wallet]`, read-only)"""
        return list(self._wallets.values())

    @property
    def exchanges(self) -> 'List[Exchange]':
        """All the exchanges in the portfolio. (`List[Exchange]`, read-only)"""
        exchanges = []
        for w in self.wallets:
            if w.exchange not in exchanges:
                exchanges += [w.exchange]
        return exchanges

    @property
    def ledger(self) -> 'Ledger':
        """The ledger that keeps track of transactions. (`Ledger`, read-only)"""
        return Wallet.ledger

    @property
    def exchange_pairs(self) -> 'List[ExchangePair]':
        """All the exchange pairs in the portfolio. (`List[ExchangePair]`, read-only)"""
        exchange_pairs = []
        for w in self.wallets:
            if w.instrument != self.base_instrument:
                exchange_pairs += [ExchangePair(w.exchange, self.base_instrument/w.instrument)]
        return exchange_pairs

    @property
    def initial_balance(self) -> 'Quantity':
        """The initial balance of the base instrument over all wallets. (`Quantity`, read-only)"""
        return self._initial_balance

    @property
    def base_balance(self) -> 'Quantity':
        """The current balance of the base instrument over all wallets. (`Quantity`, read-only)"""
        return self.balance(self.base_instrument)

    @property
    def initial_net_worth(self) -> float:
        """The initial net worth of the portfolio. (float, read-only)"""
        return self._initial_net_worth

    @property
    def net_worth(self) -> float:
        """The current net worth of the portfolio. (float, read-only)"""
        return self._net_worth

    @property
    def profit_loss(self) -> float:
        """The percent loss in net worth since the last reset. (float, read-only)"""
        return 1.0 - self.net_worth / self.initial_net_worth

    @property
    def performance(self) -> 'OrderedDict':
        """The performance of the portfolio since the last reset. (`OrderedDict`, read-only)"""
        return self._performance

    @property
    def balances(self) -> 'List[Quantity]':
        """The current unlocked balance of each instrument over all wallets. (`List[Quantity]`, read-only)"""
        return [wallet.balance for wallet in self._wallets.values()]

    @property
    def locked_balances(self) -> 'List[Quantity]':
        """The current locked balance of each instrument over all wallets. (`List[Quantity]`, read-only)"""
        return [wallet.locked_balance for wallet in self._wallets.values()]

    @property
    def total_balances(self) -> 'List[Quantity]':
        """The current total balance of each instrument over all wallets. (`List[Quantity]`, read-only)"""
        return [wallet.total_balance for wallet in self._wallets.values()]

    def balance(self, instrument: Instrument) -> 'Quantity':
        """Gets the total balance of the portfolio in a specific instrument
        available for use.

        Parameters
        ----------
        instrument : `Instrument`
            The instrument to compute the balance for.

        Returns
        -------
        `Quantity`
            The balance of the instrument over all wallets.
        """
        balance = Quantity(instrument, 0)

        for (_, symbol), wallet in self._wallets.items():
            if symbol == instrument.symbol:
                balance += wallet.balance

        return balance

    def locked_balance(self, instrument: Instrument) -> 'Quantity':
        """Gets the total balance a specific instrument locked in orders over
        the entire portfolio.

        Parameters
        ----------
        instrument : `Instrument`
            The instrument to find locked balances for.

        Returns
        -------
        `Quantity`
            The total locked balance of the instrument.
        """
        balance = Quantity(instrument, 0)

        for (_, symbol), wallet in self._wallets.items():
            if symbol == instrument.symbol:
                balance += wallet.locked_balance

        return balance

    def total_balance(self, instrument: Instrument) -> 'Quantity':
        """Gets the total balance of a specific instrument over the portfolio,
        both available for use and locked in orders.

        Parameters
        ----------
        instrument : `Instrument`
            The instrument to get total balance of.

        Returns
        -------
        `Quantity`
            The total balance of `instrument` over the portfolio.
        """
        return self.balance(instrument) + self.locked_balance(instrument)

    def get_wallet(self, exchange_id: str, instrument: 'Instrument') -> 'Wallet':
        """Gets wallet by the `exchange_id` and `instrument`.

        Parameters
        ----------
        exchange_id : str
            The exchange id used to identify the wallet.
        instrument : `Instrument`
            The instrument used to identify the wallet.

        Returns
        -------
        `Wallet`
            The wallet associated with `exchange_id` and `instrument`.
        """
        return self._wallets[(exchange_id, instrument.symbol)]

    def add(self, wallet: WalletType) -> None:
        """Adds a wallet to the portfolio.

        Parameters
        ----------
        wallet : `WalletType`
            The wallet to add to the portfolio.
        """
        if isinstance(wallet, tuple):
            wallet = Wallet.from_tuple(wallet)
        self._wallets[(wallet.exchange.id, wallet.instrument.symbol)] = wallet

    def remove(self, wallet: 'Wallet') -> None:
        """Removes a wallet from the portfolio.

        Parameters
        ----------
        wallet : `Wallet`
            The wallet to be removed.
        """
        self._wallets.pop((wallet.exchange.id, wallet.instrument.symbol), None)

    def remove_pair(self, exchange: 'Exchange', instrument: 'Instrument') -> None:
        """Removes a wallet from the portfolio by `exchange` and `instrument`.

        Parameters
        ----------
        exchange : `Exchange`
            The exchange of the wallet to be removed.
        instrument : `Instrument`
            The instrument of the wallet to be removed.
        """
        self._wallets.pop((exchange.id, instrument.symbol), None)

    @staticmethod
    def _find_keys(data: dict) -> 'List[str]':
        """Finds the keys that can attributed to the net worth of the portfolio.

        Parameters
        ----------
        data : dict
            The observer feed data point to search for keys attributed to net
            worth.

        Returns
        -------
        `List[str]`
            The list of strings attributed to net worth.
        """
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

    def on_next(self, data: dict) -> None:
        """Updates the performance metrics.

        Parameters
        ----------
        data : dict
            The data produced from the observer feed that is used to
            update the performance metrics.
        """
        data = data["internal"]

        if not self._keys:
            self._keys = self._find_keys(data)

        index = self.clock.step
        performance_data = {k: data[k] for k in self._keys}
        performance_data['base_symbol'] = self.base_instrument.symbol
        performance_step = OrderedDict()
        performance_step[index] = performance_data

        net_worth = data['net_worth']

        if self._performance is None:
            self._performance = performance_step
            self._initial_net_worth = net_worth
            self._net_worth = net_worth
        else:
            self._performance.update(performance_step)
            self._net_worth = net_worth

        if self.performance_listener:
            self.performance_listener(performance_step)

    def reset(self) -> None:
        """Resets the portfolio."""
        self._initial_balance = self.base_balance
        self._initial_net_worth = None
        self._net_worth = None
        self._performance = None

        self.ledger.reset()
        for wallet in self._wallets.values():
            wallet.reset()
