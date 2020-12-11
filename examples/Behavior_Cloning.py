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

symbol = 'AAPL'
exchange = 'NASDAQ'
start_date = '2010-01-01'
end_date = '2020-12-10'
quotes = YahooQuotesLoader.get_quotes(
        symbol=symbol,
        date_from=start_date,
        date_to=end_date,
    )
quotes.drop(columns=["index","unnamed: 0","adj close","volume"],inplace=True)
quotes.set_index("date",inplace=True)
quotes['close'].plot()


# %%

import ta
import numpy as np
# custom ta-Features
# Check https://github.com/bukosabino/ta
# Visualization: https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb
def add_custom_ta_features(
    df: pd.DataFrame,
    open: str,  # noqa
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:

    df = ta.add_volatility_ta(
        df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix
    )
    df = ta.add_trend_ta(
        df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix
    )
    df = ta.add_others_ta(df=df, close=close, fillna=fillna, colprefix=colprefix)
    return df

# get ta-indicators
quotes_ta = add_custom_ta_features(quotes,"open","high","low","close", fillna=True)

# convert to pct
#data = quotes_ta.pct_change(fill_method ='ffill')
data = quotes_ta
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(value=0)
data.reset_index(inplace=True)
data.tail()



# %%
from scipy.signal import argrelextrema
import numpy as np

# get Min/Max TimeStamps
df_min_ts = data.iloc[argrelextrema(data.close.values, np.less_equal, order=7)[0]]#['date']
df_max_ts = data.iloc[argrelextrema(data.close.values, np.greater_equal, order=7)[0]]#['date']

# %%
# Plot Min/Max with Plotly
import plotly.graph_objects as go

#fig = go.Figure(data=[go.Candlestick(x=data['date'],
#                open=data['open'],
#                high=data['high'],
#                low=data['low'],
#                close=data['close'])])
fig = go.Figure(data= go.Scatter(
    x=data['date'],
    y=data['close'],
    name='Original Data'
))
#fig = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'])])                
fig.add_trace(go.Scatter(mode="markers", x=df_min_ts['date'], y=df_min_ts['close'], name="min",marker_color='rgba(0, 255, 0, .9)'))
fig.add_trace(go.Scatter(mode="markers", x=df_max_ts['date'], y=df_max_ts['close'], name="max",marker_color='rgba(255, 0, 0, .9)'))

config = {'displayModeBar': False}
fig.show(config=config)

# %%
import matplotlib.pyplot as plt
#plt.scatter(df_min_ts.index, df_min_ts['close'], c='g')
plt.plot(df_min_ts.index, df_min_ts['close'], '^', color='green')
plt.plot(df_max_ts.index, df_max_ts['close'], 'v', color='red')
plt.plot(data['date'], data['close'], color='black')
plt.show()

# %%
features = []
#exclude date from observation - start from column 1
for c in data.columns[1:]:
    s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
    features += [s]
feed = DataFeed(features)
feed.compile()
for i in range(5):
    #print(feed.next())
    pass

# %% [markdown]
# ## Setup Trading Environment

# %%
# Make a stream of closing prices to make orders on
from tensortrade.oms.instruments import Instrument, Quantity
import tensortrade.env.default as default

def create_trade_env(data,symbol):

  # define exchange
  exchange  = Exchange("sim-exchange", service=execute_order)(
      Stream.source(list(data["close"]), dtype="float").rename(str("USD-{}").format(symbol))
  )

  # add current cash, initial-asset
  cash = Wallet(exchange, 100000 * USD)
  asset = Wallet(exchange, 0 * Instrument(symbol, 2, symbol))

  # initialize portfolio - base currency USD
  portfolio = Portfolio(
      base_instrument = USD, 
      wallets = [
          cash,
          asset
      ]
  )

  # add element for rendered feed
  renderer_feed = DataFeed([
      Stream.source(list(data["date"])).rename("date"),
      Stream.source(list(data["open"]), dtype="float").rename("open"),
      Stream.source(list(data["high"]), dtype="float").rename("high"),
      Stream.source(list(data["low"]), dtype="float").rename("low"),
      Stream.source(list(data["close"]), dtype="float").rename("close")
      #Stream.source(list(data["volume"]), dtype="float").rename("volume") 
  ])

  # define reward-scheme
  reward_scheme = default.rewards.SimpleProfit()

  # define action-scheme
  action_scheme = default.actions.BSH(
      cash=cash,
      asset=asset
  )

  # create env
  env = default.create(
      portfolio=portfolio,
      action_scheme=action_scheme,
      reward_scheme=reward_scheme,
      feed=feed,
      renderer_feed=renderer_feed,
      #renderer="screen-log",
      #window_size=20,
      max_allowed_loss=0.6
  )

  return env

env = create_trade_env(data,symbol)


# %%
env.observer.feed.next()

# %%
# ## Generate Expert Trajectories
import gym

from stable_baselines.gail import generate_expert_traj
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)
global_index = 4
global_last_action = 0

# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def expert_trader(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    global df_min_ts
    global df_max_ts
    global global_last_action

    is_buy_action = not (df_min_ts.loc[(df_min_ts['high'] == _obs[0][0]) & 
           (df_min_ts['low'] == _obs[0][1])  &
           (df_min_ts['open'] == _obs[0][2])  &
           (df_min_ts['close'] == _obs[0][3])
    ].empty)

    is_sell_action = not (df_max_ts.loc[(df_max_ts['high'] == _obs[0][0]) & 
           (df_max_ts['low'] == _obs[0][1])  &
           (df_max_ts['open'] == _obs[0][2])  &
           (df_max_ts['close'] == _obs[0][3])
        ].empty)

    if is_buy_action:
        #perform buy action
        global_last_action = 1
    elif is_sell_action:
        #perform sell action
        global_last_action = 0
    else:
        #do nothing
        pass

    return global_last_action
# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly
generate_expert_traj(expert_trader, 'expert_trader', env, n_episodes=10)
pass

# %%
# Pre-Train a Model using Behavior Cloning
from stable_baselines.gail import ExpertDataset
# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='expert_trader.npz',
                        traj_limitation=1, batch_size=128)
dataset.plot()

# %%
# export data
export_csv = df_min_ts.to_csv (r'D:\MyWork\StocksTrading\Sandboxes\AutoTrader\tensortrade\examples\data\Minimum_Points.csv', index=None, header=True)

# %%
# PPO2-Model
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
agent = PPO2(MlpPolicy, env, verbose=1)
# Pretrain the PPO2 model
agent.pretrain(dataset, n_epochs=100)

# As an option, you can train the RL agent
# model.learn(int(1e5))
agent.save(save_path=os.path.join(currentdir, "agents","BC_PPO2_MlpPolicy.zip"))


# %%
# DQN-Model
from stable_baselines.deepq.policies import MlpPolicy
agent = DQN(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(currentdir,"tf_board_log","DQN"))
agent.learn(total_timesteps=25000)
agent.save(save_path=os.path.join(currentdir, "agents","DQN_MlpPolicy_02.zip"))


# %%
# PPO2-Model
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
agent = PPO2(MlpPolicy, env, verbose=1)
agent.learn(total_timesteps=25000)
agent.save(save_path=os.path.join(currentdir, "agents","PPO2_MlpPolicy.zip"))

# %%
# PPO2-Model - VecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv


# The algorithms require a vectorized environment to run
VecEnv = DummyVecEnv([lambda: create_trade_env(data,symbol)])

agent = PPO2(MlpPolicy, env, verbose=1)
agent.learn(total_timesteps=25000)
agent.save(save_path=os.path.join(currentdir, "agents","PPO2_MlpPolicy_02_VecEnv.zip"))


# %%
# A2C-Model
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
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
def evaluate(model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  done = False
  while not done:
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)

      obs, reward, done, info = env.step(action)
      
      # Stats
      episode_rewards[-1] += reward

  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward

# %%
agent = PPO2.load(load_path=os.path.join(currentdir, "agents","PPO2_MlpPolicy.zip"))
#agent = DQN.load(load_path=os.path.join(currentdir, "agents","DQN_MlpPolicy_02.zip"), env=env)
evaluate(agent,num_steps=10000)

# %%
#portfolio.performance.net_worth.plot()
performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
performance['net_worth'].plot()


# %%
# PPO2-Model
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv


# The algorithms require a vectorized environment to run
VecEnv = DummyVecEnv([lambda: create_trade_env(data,symbol)])


