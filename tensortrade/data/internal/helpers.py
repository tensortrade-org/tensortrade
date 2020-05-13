
import numpy as np


from .wallet import create_wallet_source

from tensortrade.data import DataFeed, Stream
from tensortrade.wallets import Portfolio


def create_internal_streams(portfolio: 'Portfolio'):

    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += wallet.exchange.streams()
        sources += create_wallet_source(wallet, include_worth=(symbol != base_symbol))

    worth_streams = []
    for s in sources:
        if s.name.endswith(base_symbol + ":/total") or s.name.endswith("worth"):
            worth_streams += [s]

    net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
    sources += [net_worth]

    return sources
