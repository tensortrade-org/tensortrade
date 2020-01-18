

from tensortrade.wallets import Wallet
from tensortrade.data.stream.source import DataSource


class Balance(DataSource):

    def __init__(self, wallet: Wallet):
        super().__init__(
            name="{}_{}".format(wallet.exchange.name, wallet.instrument.symbol)
        )
        self.wallet = wallet

    def generate(self):
        return self.wallet.balance.size

    def has_next(self):
        return True

    def reset(self):
        pass


class LockedBalance(DataSource):

    def __init__(self, wallet: Wallet):
        super().__init__(
            name="{}_{}_locked".format(wallet.exchange.name, wallet.instrument.symbol)
        )
        self.wallet = wallet

    def generate(self):
        return self.wallet.locked_balance.size

    def has_next(self):
        return True

    def reset(self):
        pass
