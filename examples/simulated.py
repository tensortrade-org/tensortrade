"""
"""
import time

import numpy as np
import pandas as pd

from tensortrade.actions import DynamicOrders
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.instruments import BTC, USD, TradingPair
from tensortrade.wallets import Portfolio, Wallet


def main():
    df = pd.read_csv(
        'examples/data/Coinbase_BTCUSD_1h.csv',
        header=1,
        index_col='Date',
        usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume USD'])
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %I-%p')
    df = df.rename({'Volume USD': 'Volume'}, axis=1)
    df.columns = df.columns.str.lower()
    df = df.sort_index()

    exchange = SimulatedExchange(df)
    wallet = Wallet(exchange, .01 * BTC)
    portfolio = Portfolio(USD, wallets=[wallet])
    trading_pair = TradingPair(USD, BTC)
    action = DynamicOrders(trading_pair)
    env = TradingEnvironment(portfolio, exchange, action, 'simple',
                             window_size=20)

    times = []
    while True:
        env.step(0)
        t1 = time.time()
        env.render('human')
        times.append(time.time() - t1)
        if len(times) > 120:
            times.pop(0)
        print(f'FPS: {1 / np.mean(times):.1f}', end='\r')


if __name__ == "__main__":
    main()
