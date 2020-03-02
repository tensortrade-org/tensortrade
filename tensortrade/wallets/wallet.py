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

from typing import Dict, Tuple

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InsufficientFunds, DoubleLockedQuantity, DoubleUnlockedQuantity, QuantityNotLocked
from tensortrade.instruments import Quantity

from .ledger import Ledger


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    ledger = Ledger()

    def __init__(self, exchange: 'Exchange', quantity: 'Quantity'):
        self._exchange = exchange
        self._initial_size = quantity.size
        self._instrument = quantity.instrument
        self._balance = quantity
        self._locked = {}

    @classmethod
    def from_tuple(cls, wallet_tuple: Tuple['Exchange', 'Instrument', float]):
        exchange, instrument, balance = wallet_tuple
        return cls(exchange, Quantity(instrument, balance))

    @property
    def exchange(self) -> 'Exchange':
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'Exchange'):
        raise ValueError("You cannot change a Wallet's Exchange after initialization.")

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: 'Exchange'):
        raise ValueError("You cannot change a Wallet's Instrument after initialization.")

    @property
    def balance(self) -> 'Quantity':
        """The total balance of the wallet available for use."""
        return self._balance

    @balance.setter
    def balance(self, balance: 'Quantity'):
        self._balance = balance

    @property
    def locked_balance(self) -> 'Quantity':
        """The total balance of the wallet locked in orders."""
        locked_balance = Quantity(self.instrument, 0)

        for quantity in self._locked.values():
            locked_balance += quantity.size

        return locked_balance

    @property
    def total_balance(self) -> 'Quantity':
        """The total balance of the wallet, both available for use and locked in orders."""
        total_balance = self._balance

        for quantity in self._locked.values():
            total_balance += quantity.size

        return total_balance

    @property
    def locked(self) -> Dict[str, 'Quantity']:
        return self._locked

    def lock(self, quantity, order: 'Order', reason: str):
        if quantity.is_locked:
            raise DoubleLockedQuantity(quantity)

        if quantity > self._balance:
            raise InsufficientFunds(self.balance, quantity)

        self._balance -= quantity

        quantity = quantity.lock_for(order.path_id)

        if quantity.path_id not in self._locked:
            self._locked[quantity.path_id] = quantity
        else:
            self._locked[quantity.path_id] += quantity

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/free".format(self.exchange.name, self.instrument),
                           target="{}:{}/locked".format(self.exchange.name, self.instrument),
                           memo="LOCK ({})".format(reason))

        return quantity

    def unlock(self, quantity: 'Quantity', reason: str):
        if not quantity.is_locked:
            raise DoubleUnlockedQuantity(quantity)

        if quantity.path_id not in self._locked:
            raise QuantityNotLocked(quantity)

        if quantity > self._locked[quantity.path_id]:
            raise InsufficientFunds(self._locked[quantity.path_id], quantity)

        self._locked[quantity.path_id] -= quantity
        self._balance += quantity.free()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/locked".format(self.exchange.name, self.instrument),
                           target="{}:{}/free".format(self.exchange.name, self.instrument),
                           memo="UNLOCK {} ({})".format(self.instrument, reason))

        return quantity

    def deposit(self, quantity: 'Quantity', reason: str):
        if quantity.is_locked:
            if quantity.path_id not in self._locked:
                self._locked[quantity.path_id] = quantity
            else:
                self._locked[quantity.path_id] += quantity
        else:
            self._balance += quantity

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source=self.exchange.name,
                           target="{}:{}/locked".format(self.exchange.name, self.instrument),
                           memo="DEPOSIT ({})".format(reason))

        return quantity

    def withdraw(self, quantity: 'Quantity', reason: str):
        if quantity.is_locked and self._locked.get(quantity.path_id, False):
            if quantity > self._locked[quantity.path_id]:
                raise InsufficientFunds(self._locked[quantity.path_id], quantity)

            self._locked[quantity.path_id] -= quantity
        elif not quantity.is_locked:
            if quantity > self._balance:
                raise InsufficientFunds(self.balance, quantity)

            self._balance -= quantity

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/locked".format(self.exchange.name, self.instrument),
                           target=self.exchange.name,
                           memo="WITHDRAWAL ({})".format(reason))

        return quantity

    def reset(self):
        self._balance = Quantity(self._instrument, self._initial_size)
        self._locked = {}

    def __str__(self):
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self):
        return str(self)
