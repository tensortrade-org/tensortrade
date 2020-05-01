
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
        lambda node: node.name.endswith(base_symbol + ":/total") or node.name.endswith("worth")
    )(*sources)

    net_worth = Reduce(func=operator.add)(worth_nodes).rename("net_worth")

    sources += [net_worth]

    feed = DataFeed(sources).attach(portfolio)

    return feed
