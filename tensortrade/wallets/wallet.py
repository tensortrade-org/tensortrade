from tensortrade.base import Identifiable
from tensortrade.instruments import Quantity


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    def __init__(self, exchange: 'Exchange', quantity: 'Quantity'):
        self._exchange = exchange
        self._instrument = quantity.instrument

        self._balance = quantity
        self._locked = {}

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
        return self._balance

    @balance.setter
    def balance(self, balance: 'Quantity'):
        self._balance = balance

    @property
    def locked_balance(self) -> 'Quantity':
        return sum(self.locked.values)

    @property
    def locked(self) -> bool:
        return self._locked

    def lock_for_order(self, amount: float) -> 'Quantity':
        if self._balance < amount:
            return False

        locked_quantity = Quantity(self.instrument, amount)

        self._balance -= locked_quantity

        locked_quantity.lock_for('pending_order_id')

        self += locked_quantity

        return locked_quantity

    def __iadd__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked:
            if quantity.order_id not in self.locked.keys():
                self._locked[quantity.order_id] = quantity
            else:
                self._locked[quantity.order_id] += quantity
        else:
            self._balance += quantity

        return self

    def __isub__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked and self.locked[quantity.order_id]:
            self._locked[quantity.order_id] -= quantity
        else:
            self._balance -= quantity

        return self
