
from .wallet import Balance, LockedBalance

from tensortrade.data import BinOp
from tensortrade.wallets import Portfolio


def create_wallet_ds_from_portfolio(portfolio: Portfolio):

    sources = []
    for w in portfolio.wallets:
        available_balance = Balance(w)
        locked_balance = LockedBalance(w)
        total_balance = BinOp()
        sources += [
            Balance(w),
            LockedBalance(w),
            BinOp()
        ]
