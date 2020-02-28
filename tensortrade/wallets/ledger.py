
import pandas as pd

from typing import List
from collections import namedtuple


Transaction = namedtuple('Transaction', [
    'step',
    'exchange_name',
    'instrument',
    'operation',
    'poid',
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

    def commit(self, other):
        self._transactions += [other]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions)

    def reset(self):
        self._transactions = []