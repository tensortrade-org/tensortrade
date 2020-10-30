
from typing import List
from collections import namedtuple

import pandas as pd


Transaction = namedtuple('Transaction', [
    'poid',
    'step',
    'source',
    'target',
    'memo',
    'amount',
    'free',
    'locked',
    'locked_poid'
])


class Ledger:
    """A ledger to keep track of transactions that occur in the order
    management system."""

    def __init__(self):
        self.transactions: 'List[Transaction]' = []

    def commit(self,
               wallet: 'Wallet',
               quantity: 'Quantity',
               source: str,
               target: str,
               memo: str) -> None:
        """Commits a transaction to the ledger records.

        Parameters
        ----------
        wallet : `Wallet`
            The wallet involved in this transaction.
        quantity : `Quantity`
            The amount being used.
        source : str
            The source of funds being transferred.
        target : str
            The destination the funds are being transferred to.
        memo : str
            A description of the transaction.
        """
        poid = quantity.path_id
        locked_poid_balance = None if poid not in wallet.locked.keys() else wallet.locked[poid]

        transaction = Transaction(
            poid,
            wallet.exchange.clock.step,
            source,
            target,
            memo,
            quantity,
            wallet.balance,
            wallet.locked_balance,
            locked_poid_balance
        )

        self.transactions += [transaction]

    def as_frame(self, sort_by_order_seq: bool = False) -> 'pd.DataFrame':
        """Converts the ledger records into a data frame.

        Parameters
        ----------
        sort_by_order_seq : bool, default False
            If records should be sorted by each order path.

        Returns
        -------
        `pd.DataFrame`
            A data frame containing all the records in the ledger.
        """

        if not sort_by_order_seq:
            return pd.DataFrame(self.transactions)

        df = pd.DataFrame(self.transactions)
        frames = []
        for poid in df.poid.unique():
            frames += [df.loc[df.poid == poid, :]]

        return pd.concat(frames, ignore_index=True, axis=0)

    def reset(self):
        """Resets the ledger."""
        self.transactions = []
