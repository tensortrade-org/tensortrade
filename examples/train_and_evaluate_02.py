# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Install TensorTrade

# %%
#!python3 -m pip install git+https://github.com/tensortrade-org/tensortrade.git
import inspect
import sys
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, "{}".format(parentdir))
currentdir

# %% [markdown]
# ## Setup Data Fetching

# %%
import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH, AAPL
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger, MatplotlibTradingChart

import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines import DQN, PPO2, A2C

get_ipython().run_line_magic('matplotlib', 'inline')

# Use these commands - to reload sources, while development
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
# Use Data-Feed from Yahoo Finance
sys.path.insert(0, "{}".format("D:\MyWork\StocksTrading\Sandboxes\AutoTrader"))
from utils import YahooQuotesLoader


exchange = 'NASDAQ'
quotes = YahooQuotesLoader.get_quotes(
        symbol='AAPL',
        date_from='2010-1-1',
        date_to='2020-9-9',
    )
quotes.drop(columns=["index","unnamed: 0","adj close"],inplace=True)
quotes.set_index("date",inplace=True)
quotes['close'].plot()


# %%

import ta
import numpy as np
# get ta-indicators
quotes_ta = ta.add_all_ta_features(quotes,"open","high","low","close","volume",fillna=True)

# convert to pct
#data = quotes_ta.pct_change(fill_method ='ffill')
data = quotes
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(value=0)
data.reset_index(inplace=True)
data.tail()



# %%
features = []
for c in data.columns[1:]:
    s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
    features += [s]
feed = DataFeed(features)
feed.compile()
for i in range(5):
    print(feed.next())

# %% [markdown]
# ## Setup Trading Environment

# %%
# Make a stream of closing prices to make orders on
exchange  = Exchange("sim-exchange", service=execute_order)(
    Stream.source(list(data["close"]), dtype="float").rename("USD-AAPL")
)

portfolio = Portfolio(
    base_instrument = USD, 
    wallets = [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * AAPL)
    ]
)


renderer_feed = DataFeed([
    Stream.source(list(data["date"])).rename("date"),
    Stream.source(list(data["open"]), dtype="float").rename("open"),
    Stream.source(list(data["high"]), dtype="float").rename("high"),
    Stream.source(list(data["low"]), dtype="float").rename("low"),
    Stream.source(list(data["close"]), dtype="float").rename("close"), 
    Stream.source(list(data["volume"]), dtype="float").rename("volume") 
])


# %%
env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer_feed=renderer_feed,
    renderer="screen-log",
    window_size=20
)


# %%
env.observer.feed.next()

# %% [markdown]
# ## Setup and Train DQN Agent

# %%
from tensortrade.agents import DQNAgent
agent = DQNAgent(env)
reward = agent.train(n_steps=100, save_path="agents/", n_episodes = 100)


# %%
# DQN-Model
from stable_baselines.deepq.policies import MlpPolicy
agent = DQN(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(currentdir,"tf_board_log","DQN"))
agent.learn(total_timesteps=25000)
agent.save(save_path=os.path.join(currentdir, "agents","DQN_MlpPolicy_02.zip"))


# %%
# PPO2-Model
agent = PPO2(MlpPolicy, env, verbose=1)
agent.learn(total_timesteps=25000)
agent.save(save_path=os.path.join(currentdir, "agents","PPO2_MlpPolicy.zip"))


# %%
# A2C-Model
agent = A2C(MlpPolicy, env, verbose=1)
agent.learn(total_timesteps=25000)
agent.save(save_path=os.path.join(currentdir, "agents","A2C_MlpPolicy.zip"))


# %%
agent = DQNAgent(env)
agent.train(n_steps=200, n_episodes=2, save_path="agents/")

# %% [markdown]
# 
# %% [markdown]
# ## Evaluate Training
# 

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
portfolio.performance.plot()


# %%
portfolio.performance.net_worth.plot()

# %% [markdown]
# ### Environment with Multiple Renderers
# Create PlotlyTradingChart and FileLogger renderers. Configuring renderers is optional as they can be used with their default settings.

# %%
chart_renderer = PlotlyTradingChart(
    display=True,  # show the chart on screen (default)
    #height=800,  # affects both displayed and saved file height. None for 100% height.
    #save_format="html",  # save the chart to an HTML file
    #auto_open_html=True,  # open the saved HTML chart in a new browser tab
)

file_logger = FileLogger(
    filename="dqn_test.log",  # omit or None for automatic file name
    path= os.path.join(currentdir, "test_logs")  # create a new directory if doesn't exist, None for no directory
)


# %%
# custom env
env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    window_size=20,
    renderer_feed=feed,
    renderers=[
        chart_renderer, 
        file_logger
    ]
)


# %%
agent = PPO2.load(load_path=os.path.join(currentdir, "agents","MlpPolicy_02.zip"), env=env)
#agent = DQN.load(load_path=os.path.join(currentdir, "agents","DQN_MlpPolicy_02.zip"), env=env)
done = False
obs = env.reset()
count = 0
while not done:
    action, _states = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    #count += 1
    #done = count > 1000


# %%
portfolio.performance.net_worth.plot()


# %%



