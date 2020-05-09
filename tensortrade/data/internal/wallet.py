
import operator

from tensortrade.data import Lambda, Module, Select, BinOp
from tensortrade.wallets import Wallet


def create_wallet_source(wallet: Wallet, include_worth=True):
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    with Module(exchange_name + ":/" + symbol) as wallet_ds:
        free_balance = Lambda(lambda w: w.balance.as_float(), wallet, name="free")
        locked_balance = Lambda(lambda w: w.locked_balance.as_float(), wallet, name="locked")
        total_balance = Lambda(lambda w: w.total_balance.as_float(), wallet, name="total")

        nodes = [free_balance, locked_balance, total_balance]

        if include_worth:
            price = Select(lambda node: node.name.endswith(symbol))(wallet.exchange)
            worth = BinOp(operator.mul, name="worth")(price, total_balance)
            nodes += [worth]

    return wallet_ds
