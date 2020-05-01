import ssl

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathlib import Path

from tensortrade.agents.parallel_dqn.parallel_dqn_agent import ParallelDQNAgent

from tensortrade.data import DataFeed, Stream, Module
from tensortrade.environments import TradingEnvironment
from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.instruments import USD, BTC, ETH
from tensortrade.wallets import Portfolio, Wallet

def create_env():
    def fetch_data(exchange_name, symbol, timeframe):
        url = "https://www.cryptodatadownload.com/cdd/"
        filename = "{}_{}USD_{}.csv".format(exchange_name, symbol, timeframe)
        volume_column = "Volume {}".format(symbol)
        new_volume_column = "Volume_{}".format(symbol)

        df = pd.read_csv(url + filename, skiprows=1)
        df = df[::-1]
        df = df.drop(["Symbol"], axis=1)
        df = df.rename({"Volume USD": "volume", volume_column: new_volume_column}, axis=1)
        df = df.set_index("Date")
        df.columns = [symbol + ":" + name.lower() for name in df.columns]

        return df

    ssl._create_default_https_context = ssl._create_unverified_context  # Only used if pandas gives a SSLError
    coinbase_data = pd.concat([
        fetch_data("Coinbase", "BTC", "1h"),
        fetch_data("Coinbase", "ETH", "1h")
    ], axis=1)

    coinbase = Exchange("coinbase", service=execute_order)(
        Stream("USD-BTC", list(coinbase_data['BTC:close'])),
        Stream("USD-ETH", list(coinbase_data['ETH:close']))
    )

    with Module("coinbase") as coinbase_ns:
        nodes = [Stream(name, list(coinbase_data[name])) for name in coinbase_data.columns]

    feed = DataFeed([coinbase_ns])

    portfolio = Portfolio(USD, [
        Wallet(coinbase, 10000 * USD),
        Wallet(coinbase, 10 * BTC),
        Wallet(coinbase, 5 * ETH),
    ])

    env = TradingEnvironment(
        feed=feed,
        portfolio=portfolio,
        action_scheme='managed-risk',
        reward_scheme='risk-adjusted',
        window_size=20
    )

    return env


if __name__ == '__main__':

    parallel = True

    save_path = "agents/"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    if parallel:
        create_env_func = create_env

        agent = ParallelDQNAgent(create_env_func)

        # IMPORTANT
        # it is best leave 1 core free to avoid any process desync issues and leave processing power to the optimizer
        # which is running on the main thread
        #
        # if you see that the processes don't finish at the same time reduce n_envs,
        # this can cause the training to slow down because finished processes will wait for slower processes
        # if you have a strong enough cpu to handle more than one env per core than you can try increasing n_envs
        #
        # more processes don't always lead to dramatically better/faster results
        # you should treat n_envs as another hyperparameter

        reward, test_env = agent.train(n_envs=mp.cpu_count() - 1, n_steps=500,
                                       n_episodes=10, save_path=save_path,
                                       batch_size=1000, test_model=True,
                                       memory_capacity=5000)

        test_env.portfolio.performance.net_worth.plot()
        plt.show()

    else:
        from tensortrade.agents import DQNAgent

        env = create_env()

        agent = DQNAgent(env)

        reward = agent.train(n_steps=200, n_episodes=25, save_path=save_path)

        env.portfolio.performance.net_worth.plot()
        plt.show()
        print("P/L: {}".format(env.portfolio.profit_loss))
        print("mean_reward: {}".format(reward))
