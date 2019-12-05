from . import Quantity


class Wallet:
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    def __init__(self, exchange: 'Exchange', instrument: 'Instrument', balance: float = 0):
        self.exchange = exchange
        self.instrument = instrument
        self._balance = Quantity(balance, instrument)

        # self.exchange.attach_wallet(self)

        self._locked = {}

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, balance: 'Quantity'):
        self._balance = balance

    @property
    def locked(self):
        return self._locked

    def __iadd__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked():
            if quantity.order_id not in self.locked.keys():
                self.locked[quantity.order_id] = quantity
            else:
                self.locked[quantity.order_id] += quantity
        else:
            self._balance += quantity
        return self

    def __isub__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked() and self.locked[quantity.order_id]:
            self.locked[quantity.order_id] -= quantity
        else:
            self._balance -= quantity
        return self
