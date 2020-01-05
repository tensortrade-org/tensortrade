import time

from tensortrade.actions import DynamicOrders
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.live import CCXTExchange
from tensortrade.instruments import BTC, USD, TradingPair
from tensortrade.wallets import Portfolio, Wallet


def main():
    trading_pair = TradingPair(USD, BTC)
    action = DynamicOrders(trading_pair)
    exchange = CCXTExchange('bitmex', observation_pairs=[trading_pair],
                            timeframe='1m')
    wallet = Wallet(exchange, .01 * BTC)
    portfolio = Portfolio(USD, wallets=[wallet])
    env = TradingEnvironment(portfolio, exchange, action, 'simple')
    while True:
        env.step(0)
        env.render('chart')
        time.sleep(1)


if __name__ == "__main__":
    main()
