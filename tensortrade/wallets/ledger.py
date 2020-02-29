
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
    'locked'
])


class Ledger:

    def __init__(self):
        self._transactions = []

    @property
    def transactions(self) -> List['Transaction']:
        return self._transactions

    def commit(self, wallet: 'Wallet', quantity: 'Quantity', source: str, target: str, memo: str):
        transaction = Transaction(
            quantity.path_id,
            wallet.exchange.clock.step,
            source,
            target,
            memo,
            quantity,
            wallet.balance,
            wallet.locked_balance
        )

        self._transactions += [transaction]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions)

    def reset(self):
        self._transactions = []
