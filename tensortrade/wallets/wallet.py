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

import numpy as np

from typing import Dict, Tuple
from collections import namedtuple
from decimal import Decimal

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InsufficientFunds, DoubleLockedQuantity, DoubleUnlockedQuantity, QuantityNotLocked
from tensortrade.instruments import Quantity, ExchangePair

from .ledger import Ledger


Transfer = namedtuple("Transfer", ["quantity", "commission", "price"])


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    ledger = Ledger()

    def __init__(self, exchange: 'Exchange', quantity: 'Quantity'):
        self._exchange = exchange
        self._initial_size = quantity.size
        self._instrument = quantity.instrument
        self._balance = quantity.quantize()
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

        self._locked[quantity.path_id] = self._locked[quantity.path_id].quantize()
        self._balance = self._balance.quantize()

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

        self._locked[quantity.path_id] = self._locked[quantity.path_id].quantize()
        self._balance = self._balance.quantize()

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

        self._balance = self._balance.quantize()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source=self.exchange.name,
                           target="{}:{}/locked".format(self.exchange.name, self.instrument),
                           memo="DEPOSIT ({})".format(reason))

        return quantity

    def withdraw(self, quantity: 'Quantity', reason: str):
        if quantity.is_locked and self._locked.get(quantity.path_id, False):
            locked_quantity = self._locked[quantity.path_id]
            if quantity > locked_quantity:
                raise InsufficientFunds(self._locked[quantity.path_id], quantity)
            self._locked[quantity.path_id] -= quantity

        elif not quantity.is_locked:
            if quantity > self._balance:
                raise InsufficientFunds(self.balance, quantity)
            self._balance -= quantity

        self._balance = self._balance.quantize()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/locked".format(self.exchange.name, self.instrument),
                           target=self.exchange.name,
                           memo="WITHDRAWAL ({})".format(reason))

        return quantity

    @staticmethod
    def transfer(source: 'Wallet',
                 target: 'Wallet',
                 quantity: 'Quantity',
                 commission: 'Quantity',
                 exchange_pair: 'ExchangePair',
                 reason: str):
        """
        Parameters
        ----------
            source : 'Wallet'
                The wallet in which funds will be transferred from
            target : 'Wallet'
                The wallet in which funds will be transferred to
            quantity : 'Quantity'
                The quantity to be transferred from the source to the target.
                In terms of the instrument of the source wallet.
            commission :  'Quantity'
                The commission to be taken from the source wallet for performing
                the transfer of funds.
            exchange_pair : 'ExchangePair'
                The exchange pair associated with the transfer
        """
        quantity = quantity.quantize()
        commission = commission.quantize()

        pair = source.instrument / target.instrument
        poid = quantity.path_id

        lsb1 = source.locked.get(poid).size
        ltb1 = target.locked.get(poid, 0 * pair.quote).size

        commission = source.withdraw(commission, "COMMISSION")
        quantity = source.withdraw(quantity, "FILL ORDER")

        if quantity.instrument == exchange_pair.pair.base:
            instrument = exchange_pair.pair.quote
            converted_size = quantity.size / exchange_pair.price
        else:
            instrument = exchange_pair.pair.base
            converted_size = quantity.size * exchange_pair.price

        converted = Quantity(instrument, converted_size, quantity.path_id).quantize()

        converted = target.deposit(converted, 'TRADED {} {} @ {}'.format(quantity,
                                                                         exchange_pair,
                                                                         exchange_pair.price))

        lsb2 = source.locked.get(poid).size
        ltb2 = target.locked.get(poid, 0 * pair.quote).size

        q = quantity.size
        c = commission.size
        cv = converted.size
        p = exchange_pair.inverse_price if pair == exchange_pair.pair else exchange_pair.price

        source_quantization = Decimal(10) ** -source.instrument.precision
        target_quantization = Decimal(10) ** -target.instrument.precision

        lhs = Decimal((lsb1 - lsb2) - (q + c)).quantize(source_quantization)
        rhs = Decimal(ltb2 - ltb1 - cv).quantize(target_quantization)

        lhs_eq_zero = np.isclose(float(lhs), 0, atol=float(source_quantization))
        rhs_eq_zero = np.isclose(float(rhs), 0, atol=float(target_quantization))

        if not lhs_eq_zero or not rhs_eq_zero:
            equation = "({} - {}) - ({} + {}) != ({} - {}) - {}   [LHS = {}, RHS = {}, Price = {}]".format(
                lsb1, lsb2, q, c, ltb2, ltb1, cv, lhs, rhs, p
            )

            raise Exception("Invalid Transfer: " + equation)

        return Transfer(quantity, commission, exchange_pair.price)

    def reset(self):
        self._balance = Quantity(self._instrument, self._initial_size).quantize()
        self._locked = {}

    def __str__(self):
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self):
        return str(self)
