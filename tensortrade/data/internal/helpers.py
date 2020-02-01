
import operator


from .wallet import create_wallet_source

from tensortrade.data import DataFeed, Reduce, Condition
from tensortrade.wallets import Portfolio


def create_internal_feed(portfolio: 'Portfolio'):

    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += [wallet.exchange]
        sources += [create_wallet_source(wallet, include_worth=(symbol != base_symbol))]

    worth_nodes = Condition(
        "worths",
        lambda node: node.name.endswith(base_symbol + ":/total") or node.name.endswith("worth")
    )(*sources)

    net_worth = Reduce("net_worth", func=operator.add)(worth_nodes)

    sources += [net_worth]

    feed = DataFeed(sources)
    feed.attach(portfolio)

    return feed
