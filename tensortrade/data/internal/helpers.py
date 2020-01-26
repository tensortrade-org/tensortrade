
import operator


from .wallet import create_wallet_source

from tensortrade.data import DataFeed, Reduce
from tensortrade.wallets import Portfolio


def create_internal_feed(portfolio: 'Portfolio'):

    base_symbol = portfolio.base_instrument.symbol

    sources = []
    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += [wallet.exchange]
        sources += [create_wallet_source(wallet, include_worth=(symbol != base_symbol))]

    net_worth = Reduce(
        name="net_worth",
        selector=lambda k: k.endswith(base_symbol + ":/total") or k.endswith("worth"),
        func=operator.add
    )(*sources)

    sources += [net_worth]

    feed = DataFeed(sources)
    feed.attach(portfolio)
    return feed
