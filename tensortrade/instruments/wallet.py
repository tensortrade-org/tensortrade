from . import Quantity


class Wallet():
    """A wallet containing financial instruments to be traded on a specific exchange."""

    def __init__(self, exchange: 'Exchange'):
        self.exchange = exchange

        self.exchange.attach_wallet(self)

        self._quantities = {}
        self._locked_quantities = {}

    @property
    def quantities(self):
        return self._quantities

    @property
    def locked_quantities(self):
        return self._locked_quantities

    def lock(self, quantity: 'Quantity'):
        if not self.has(quantity):
            return False

        self.remove(quantity)
        self.add_locked(quantity)

        return True

    def unlock(self, quantity: 'Quantity'):
        if not self.has_locked(quantity):
            return False

        self.remove_locked(quantity)
        self.add(quantity)

        return True

    def has(self, quantity: 'Quantity'):
        amount = self._quantities[quantity.instrument] or 0
        return amount > quantity.amount

    def has_locked(self, quantity: 'Quantity'):
        amount = self._locked_quantities[quantity.instrument] or 0
        return amount > quantity.amount

    def add(self, quantity: 'Quantity'):
        return self._add_quantity(quantity, locked=False)

    def remove(self, quantity: 'Quantity') -> 'Quantity':
        return self._remove_quantity(quantity, locked=False)

    def add_locked(self, quantity: 'Quantity'):
        return self._add_quantity(quantity, locked=True)

    def remove_locked(self, quantity: 'Quantity') -> 'Quantity':
        return self._remove_quantity(quantity, locked=True)

    def _add_quantity(self, quantity: 'Quantity', locked=False):
        quantities = self._locked_quantities if locked else self._quantities

        if quantities[quantity.instrument]:
            quantities[quantity.instrument] += quantity.amount
        else:
            quantities[quantity.instrument] = quantity.amount

    def _remove_quantity(self, quantity: 'Quantity', locked=False) -> 'Quantity':
        quantities = self._locked_quantities if locked else self._quantities

        if (not locked and self.has(quantity)) or (locked and self.has_locked(quantity)):
            quantities[quantity.instrument] -= quantity.amount
            return quantity.amount

        amount = quantities[quantity.instrument] or 0

        quantities[quantity.instrument] = 0

        return Quantity(quantity.instrument, amount)
