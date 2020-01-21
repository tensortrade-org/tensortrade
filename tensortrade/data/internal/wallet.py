
import operator

from tensortrade.data import Lambda, Namespace, Select, BinOp
from tensortrade.wallets import Wallet


def create_wallet_source(wallet: Wallet, include_worth=True):
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    free_balance = Lambda("free", lambda w: w.balance.size, wallet)
    locked_balance = Lambda("locked", lambda w: w.locked_balance.size, wallet)
    total_balance = Lambda("total", lambda w: w.total_balance.size, wallet)

    if include_worth:
        price = Select(lambda k: k.endswith(symbol))(wallet.exchange)
        worth = BinOp("worth", operator.mul)(price, total_balance)
        nodes = [free_balance, locked_balance, total_balance, worth]
    else:
        nodes = [free_balance, locked_balance, total_balance]

    name = exchange_name + ":/" + symbol
    wallet_ds = Namespace(name)(*nodes)

    return wallet_ds
