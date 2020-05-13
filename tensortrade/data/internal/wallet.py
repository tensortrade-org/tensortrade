
import operator

from tensortrade.data import Stream, NameSpace
from tensortrade.wallets import Wallet


def create_wallet_source(wallet: Wallet, include_worth=True):
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    streams = []

    with NameSpace(exchange_name + ":/" + symbol):
        free_balance = Stream.sensor(wallet, lambda w: w.balance.as_float(), dtype="float").rename("free")
        locked_balance = Stream.sensor(wallet, lambda w: w.locked_balance.as_float(), dtype="float").rename("locked")
        total_balance = Stream.sensor(wallet, lambda w: w.total_balance.as_float(), dtype="float").rename("total")

        streams += [free_balance, locked_balance, total_balance]

        if include_worth:
            price = Stream.select(wallet.exchange.streams(), lambda node: node.name.endswith(symbol))
            worth = (price * total_balance).rename("worth")
            streams += [worth]

    return streams
