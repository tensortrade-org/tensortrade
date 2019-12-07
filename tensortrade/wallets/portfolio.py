import pandas as pd

from typing import Union, List, Dict

from tensortrade import Component


class Portfolio(Component):
    """A portfolio of wallets for use on any ."""

    registered_name = "portfolio"

    def __init__(self):
        self._base = self.context.base
        self._wallets = {}

    @property
    def base(self) -> 'Instrument':
        """The exchange instrument used to measure value and performance statistics."""
        return self._base

    @base.setter
    def base(self, base: 'Instrument'):
        self._base = base

    @property
    def initial_balance(self) -> float:
        """The initial balance of the base instrument over all wallets."""
        raise NotImplementedError

    @property
    def balance(self) -> float:
        """The current balance of the base instrument over all wallets."""
        raise NotImplementedError

    @property
    def balances(self) -> Dict['Instrument', float]:
        """The current balance of each instrument over all exchanges (non-positive balances excluded)."""
        raise NotImplementedError

    @property
    def performance(self) -> pd.DataFrame:
        """The performance of the active account on the exchange since the last reset.

        A pandas dataframe with the following columns:

            | step | balance | net_worth |
        """
        raise NotImplementedError

    @property
    def instrument_balance(self, symbol: str) -> float:
        """The current balance of the specified symbol on the exchange, denoted in the base instrument.

        Arguments:
            symbol: The symbol to retrieve the balance of.

        Returns:
            The balance of the specified exchange symbol, denoted in the base instrument.
        """
        wallets = self._wallets

        if symbol in wallets.keys():
            return wallets[symbol]

        return 0

    def group_by_exchange(self):
        pass

    def filter_by(self, function):
        return list(filter(self._wallets, function))

    @property
    def net_worth(self) -> float:
        """Calculate the net worth of the active account on the exchange.

        Returns:
            The total portfolio value of the active account on the exchange.
        """
        # %%TODO: Change this method to compute over all exchanges and instruments.
        net_worth = self.balance
        wallets = self._wallets

        if not wallets:
            return net_worth

        for (exchange, instrument), wallet in wallets.items():
            if instrument == self._base:
                net_worth += wallet.balance

            current_price = self._exchange.quote_price(instrument=instrument)
            net_worth += current_price * wallet.balance

        return net_worth

    @property
    def profit_loss_percent(self) -> float:
        """Calculate the percentage change in net worth since the last reset.

        Returns:
            The percentage change in net worth since the last reset.
        """
        raise NotImplemented()

    def get_wallet(self, exchange_id: str, instrument: 'Instrument'):
        return self._wallets[(exchange_id, instrument)]

    def add(self, wallet: 'Wallet'):
        k = (str(wallet.exchange.id), wallet.instrument)
        self._wallets[k] = wallet

    def remove(self, wallet: 'Wallet'):
        del self._wallets[(str(wallet.exchange.id), wallet.instrument)]