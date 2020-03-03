
import pandas as pd

from typing import List
from collections import namedtuple


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

    def __init__(self):
        self._transactions = []

    @property
    def transactions(self) -> List['Transaction']:
        return self._transactions

    def commit(self, wallet: 'Wallet', quantity: 'Quantity', source: str, target: str, memo: str):

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

        self._transactions += [transaction]

    def as_frame(self, sort_by_order_seq=False) -> pd.DataFrame:

        if not sort_by_order_seq:
            return pd.DataFrame(self.transactions)

        df = pd.DataFrame(self.transactions)
        frames = []
        for poid in df.poid.unique():
            frames += [df.loc[df.poid == poid, :]]

        return pd.concat(frames, ignore_index=True, axis=0)

    def reset(self):
        self._transactions = []
