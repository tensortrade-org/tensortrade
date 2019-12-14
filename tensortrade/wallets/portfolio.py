import pandas as pd
import numpy as np

from typing import Callable, Tuple, Union, List, Dict, NewType

from tensortrade import Component, TimedIdentifiable
from tensortrade.instruments import Instrument, Quantity, TradingPair

from .wallet import Wallet


WalletType = Union['Wallet', Tuple['Exchange', Instrument, float]]


class Portfolio(Component, TimedIdentifiable):
    """A portfolio of wallets on exchanges."""

    registered_name = "portfolio"

    def __init__(self,
                 base_instrument: Instrument,
                 wallets: List[WalletType] = None):
        wallets = wallets or []

        self._base_instrument = self.default('base_instrument', base_instrument)
        self._wallets = {}

        for wallet in wallets:
            self.add(wallet)

        self._initial_balance = self.base_balance
        self._initial_net_worth = self.net_worth
        self._performance = pd.DataFrame([], columns=['step', 'net_worth'], index=['step'])

    @property
    def base_instrument(self) -> Instrument:
        """The exchange instrument used to measure value and performance statistics."""
        return self._base_instrument

    @base_instrument.setter
    def base_instrument(self, base_instrument: Instrument):
        self._base_instrument = base_instrument

    @property
    def wallets(self) -> List[Wallet]:
        return list(self._wallets.values())

    @property
    def initial_balance(self) -> Quantity:
        """The initial balance of the base instrument over all wallets, set by calling `reset`."""
        return self._initial_balance

    @property
    def base_balance(self) -> Quantity:
        """The current balance of the base instrument over all wallets."""
        return self.balance(self._base_instrument)

    @property
    def net_worth(self) -> float:
        """Calculate the net worth of the active account on the exchange.

        Returns:
            The total portfolio value of the active account on the exchange.
        """
        net_worth = 0

        if not self._wallets:
            return net_worth

        for wallet in self._wallets.values():
            if wallet.instrument == self._base_instrument:
                current_price = 1
            else:
                pair = TradingPair(self._base_instrument, wallet.instrument)
                current_price = wallet.exchange.quote_price(pair)

            wallet_balance = wallet.total_balance.size
            net_worth += current_price * wallet_balance

        return net_worth

    @property
    def profit_loss(self) -> float:
        """Calculate the percentage change in net worth since the last reset.

        Returns:
            The percentage change in net worth since the last reset.
            i.e. A return value of 2 would indicate a 100% increase in net worth (e.g. $100 -> $200)
        """
        return self.net_worth / self._initial_net_worth

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

    def get_wallet(self, exchange_id: str, instrument: Instrument):
        return self._wallets[(exchange_id, instrument.symbol)]

    def add(self, wallet: WalletType):
        if isinstance(wallet, tuple):
            wallet = Wallet.from_tuple(wallet)
        k = (wallet.exchange.id, wallet.instrument.symbol)
        self._wallets[k] = wallet

    def remove(self, wallet: 'Wallet'):
        self._wallets.pop((wallet.exchange.id, wallet.instrument.symbol), None)

    def remove_pair(self, exchange: 'Exchange', instrument: Instrument):
        self._wallets.pop((exchange.id, instrument.symbol), None)

    def update(self):

        columns = ['step', 'net_worth']
        names = [[w.instrument.symbol, "{}_pending".format(w.instrument.symbol)] for w in self.wallets]
        columns += list(np.array(names).flatten())

        data = [self.clock.step, self.net_worth]
        locked_data = [[w.balance.size, w.locked_balance.size] for w in self.wallets]
        data += list(np.array(locked_data).flatten())

        performance_update = pd.DataFrame(data, columns=columns)

        self._performance = pd.concat([self._performance, performance_update], axis=0, sort=True).dropna()

    def reset(self):
        self._initial_balance = self.base_balance
        self._initial_net_worth = self.net_worth
        self._performance = pd.DataFrame([], columns=['step', 'net_worth'], index=['step'])
