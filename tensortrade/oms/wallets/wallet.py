# Copyright 2020 The TensorTrade Authors.
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
from collections import namedtuple
from decimal import Decimal

import numpy as np

from tensortrade.core import Identifiable
from tensortrade.core.exceptions import (
    InsufficientFunds,
    DoubleLockedQuantity,
    DoubleUnlockedQuantity,
    QuantityNotLocked
)
from tensortrade.oms.instruments import Instrument, Quantity, ExchangePair
from tensortrade.oms.orders import Order
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.wallets.ledger import Ledger


Transfer = namedtuple("Transfer", ["quantity", "commission", "price"])


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange.

    Parameters
    ----------
    exchange : `Exchange`
        The exchange associated with this wallet.
    balance : `Quantity`
        The initial balance quantity for the wallet.
    """

    ledger = Ledger()

    def __init__(self, exchange: 'Exchange', balance: 'Quantity'):
        self.exchange = exchange
        self._initial_size = balance.size
        self.instrument = balance.instrument
        self.balance = balance.quantize()
        self._locked = {}

    @property
    def locked_balance(self) -> 'Quantity':
        """The total balance of the wallet locked in orders. (`Quantity`, read-only)"""
        locked_balance = Quantity(self.instrument, 0)

        for quantity in self._locked.values():
            locked_balance += quantity.size

        return locked_balance

    @property
    def total_balance(self) -> 'Quantity':
        """The total balance of the wallet available for use and locked in orders. (`Quantity`, read-only)"""
        total_balance = self.balance

        for quantity in self._locked.values():
            total_balance += quantity.size

        return total_balance

    @property
    def locked(self) -> 'Dict[str, Quantity]':
        """The current quantities that are locked for orders. (`Dict[str, Quantity]`, read-only)"""
        return self._locked

    def lock(self, quantity, order: 'Order', reason: str) -> 'Quantity':
        """Locks funds for specified order.

        Parameters
        ----------
        quantity : `Quantity`
            The amount of funds to lock for the order.
        order : `Order`
            The order funds will be locked for.
        reason : str
            The reason for locking funds.

        Returns
        -------
        `Quantity`
            The locked quantity for `order`.

        Raises
        ------
        DoubleLockedQuantity
            Raised if the given amount is already a locked quantity.
        InsufficientFunds
            Raised if amount is greater the current balance.
        """
        if quantity.is_locked:
            raise DoubleLockedQuantity(quantity)

        if quantity > self.balance:
            if (quantity-self.balance)>Decimal(10)**(-self.instrument.precision+2):
                raise InsufficientFunds(self.balance, quantity)
            else:
                quantity = self.balance

        self.balance -= quantity

        quantity = quantity.lock_for(order.path_id)

        if quantity.path_id not in self._locked:
            self._locked[quantity.path_id] = quantity
        else:
            self._locked[quantity.path_id] += quantity

        self._locked[quantity.path_id] = self._locked[quantity.path_id].quantize()
        self.balance = self.balance.quantize()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/free".format(self.exchange.name, self.instrument),
                           target="{}:{}/locked".format(self.exchange.name, self.instrument),
                           memo="LOCK ({})".format(reason))

        return quantity

    def unlock(self, quantity: 'Quantity', reason: str) -> 'Quantity':
        """Unlocks a certain amount from the locked funds of the wallet that
        are associated with the given `quantity` path id.

        Parameters
        ----------
        quantity : `Quantity`
            The quantity to unlock from the funds.
        reason : str
            The reason for unlocking funds.

        Returns
        -------
        `Quantity`
            The free quantity.

        Raises
        ------
        DoubleUnlockedFunds
            Raised if `quantity` is not a locked quantity.
        QuantityNotLocked
            Raised if  `quantity` has a path id that is not currently allocated
            in this wallet.
        InsufficientFunds
            Raised if `quantity` is greater than the amount currently allocated
            for the associated path id.
        """
        if not quantity.is_locked:
            raise DoubleUnlockedQuantity(quantity)

        if quantity.path_id not in self._locked:
            raise QuantityNotLocked(quantity)

        if quantity > self._locked[quantity.path_id]:
            raise InsufficientFunds(self._locked[quantity.path_id], quantity)

        self._locked[quantity.path_id] -= quantity
        self.balance += quantity.free()

        self._locked[quantity.path_id] = self._locked[quantity.path_id].quantize()
        self.balance = self.balance.quantize()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/locked".format(self.exchange.name, self.instrument),
                           target="{}:{}/free".format(self.exchange.name, self.instrument),
                           memo="UNLOCK {} ({})".format(self.instrument, reason))

        return quantity

    def deposit(self, quantity: 'Quantity', reason: str) -> 'Quantity':
        """Deposits funds into the wallet.

        Parameters
        ----------
        quantity : `Quantity`
            The amount to deposit into this wallet.
        reason : str
            The reason for depositing the amount.

        Returns
        -------
        `Quantity`
            The deposited amount.
        """
        if quantity.is_locked:
            if quantity.path_id not in self._locked:
                self._locked[quantity.path_id] = quantity
            else:
                self._locked[quantity.path_id] += quantity
        else:
            self.balance += quantity

        self.balance = self.balance.quantize()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source=self.exchange.name,
                           target="{}:{}/locked".format(self.exchange.name, self.instrument),
                           memo="DEPOSIT ({})".format(reason))

        return quantity

    def withdraw(self, quantity: 'Quantity', reason: str) -> 'Quantity':
        """Withdraws funds from the wallet.

        Parameters
        ----------
        quantity : `Quantity`
            The amount to withdraw from this wallet.
        reason : str
            The reason for withdrawing the amount.

        Returns
        -------
        `Quantity`
            The withdrawn amount.
        """
        if quantity.is_locked and self._locked.get(quantity.path_id, False):
            locked_quantity = self._locked[quantity.path_id]
            if quantity > locked_quantity:
                if (quantity-locked_quantity)>Decimal(10)**(-self.instrument.precision+2):
                    raise InsufficientFunds(locked_quantity, quantity)
                else:
                    quantity = locked_quantity
            self._locked[quantity.path_id] -= quantity

        elif not quantity.is_locked:
            if quantity > self.balance:
                if (quantity-self.balance)>Decimal(10)**(-self.instrument.precision+2):
                    raise InsufficientFunds(self.balance, quantity)
                else:
                    quantity = self.balance
            self.balance -= quantity

        self.balance = self.balance.quantize()

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/locked".format(self.exchange.name, self.instrument),
                           target=self.exchange.name,
                           memo="WITHDRAWAL ({})".format(reason))

        return quantity

    @classmethod
    def from_tuple(cls, wallet_tuple: 'Tuple[Exchange, Instrument, float]') -> 'Wallet':
        """Creates a wallet from a wallet tuple.

        Parameters
        ----------
        wallet_tuple : `Tuple[Exchange, Instrument, float]`
            A tuple containing an exchange, instrument, and amount.

        Returns
        -------
        `Wallet`
            A wallet corresponding to the arguments given in the tuple.
        """
        exchange, instrument, balance = wallet_tuple
        return cls(exchange, Quantity(instrument, balance))

    @staticmethod
    def transfer(source: 'Wallet',
                 target: 'Wallet',
                 quantity: 'Quantity',
                 commission: 'Quantity',
                 exchange_pair: 'ExchangePair',
                 reason: str) -> 'Transfer':
        """Transfers funds from one wallet to another.

        Parameters
        ----------
        source : `Wallet`
            The wallet in which funds will be transferred from
        target : `Wallet`
            The wallet in which funds will be transferred to
        quantity : `Quantity`
            The quantity to be transferred from the source to the target.
            In terms of the instrument of the source wallet.
        commission :  `Quantity`
            The commission to be taken from the source wallet for performing
            the transfer of funds.
        exchange_pair : `ExchangePair`
            The exchange pair associated with the transfer
        reason : str
            The reason for transferring the funds.

        Returns
        -------
        `Transfer`
            A transfer object describing the transaction.

        Raises
        ------
        Exception
            Raised if an equation that describes the conservation of funds
            is broken.
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

    def reset(self) -> None:
        """Resets the wallet."""
        self.balance = Quantity(self.instrument, self._initial_size).quantize()
        self._locked = {}

    def __str__(self) -> str:
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self) -> str:
        return str(self)
