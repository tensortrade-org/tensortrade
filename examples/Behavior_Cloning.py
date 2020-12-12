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
import queue
import time
from multiprocessing import Queue, Process

import cv2  # pytype:disable=import-error
import numpy as np
from joblib import Parallel, delayed

from stable_baselines import logger


class ExpertDataset(object):
    """
    Dataset for using behavior cloning or GAIL.

    The structure of the expert dataset is a dict, saved as an ".npz" archive.
    The dictionary contains the keys 'actions', 'episode_returns', 'rewards', 'obs' and 'episode_starts'.
    The corresponding values have data concatenated across episode: the first axis is the timestep,
    the remaining axes index into the data. In case of images, 'obs' contains the relative path to
    the images, to enable space saving from image compression.

    :param expert_path: (str) The path to trajectory data (.npz file). Mutually exclusive with traj_data.
    :param traj_data: (dict) Trajectory data, in format described above. Mutually exclusive with expert_path.
    :param train_fraction: (float) the train validation split (0 to 1)
        for pre-training using behavior cloning (BC)
    :param batch_size: (int) the minibatch size for behavior cloning
    :param traj_limitation: (int) the number of trajectory to use (if -1, load all)
    :param randomize: (bool) if the dataset should be shuffled
    :param verbose: (int) Verbosity
    :param sequential_preprocessing: (bool) Do not use subprocess to preprocess
        the data (slower but use less memory for the CI)
    """
    # Excluded attribute when pickling the object
    EXCLUDED_KEYS = {'dataloader', 'train_loader', 'val_loader'}

    def __init__(self, expert_path=None, traj_data=None, train_fraction=0.7, batch_size=64,
                 traj_limitation=-1, randomize=True, verbose=1, sequential_preprocessing=False):
        if traj_data is not None and expert_path is not None:
            raise ValueError("Cannot specify both 'traj_data' and 'expert_path'")
        if traj_data is None and expert_path is None:
            raise ValueError("Must specify one of 'traj_data' or 'expert_path'")
        if traj_data is None:
            traj_data = np.load(expert_path, allow_pickle=True)

        if verbose > 0:
            for key, val in traj_data.items():
                print(key, val.shape)

        # Array of bool where episode_starts[i] = True for each new episode
        episode_starts = traj_data['episode_starts']

        traj_limit_idx = len(traj_data['obs'])

        if traj_limitation > 0:
            n_episodes = 0
            # Retrieve the index corresponding
            # to the traj_limitation trajectory
            for idx, episode_start in enumerate(episode_starts):
                n_episodes += int(episode_start)
                if n_episodes == (traj_limitation + 1):
                    traj_limit_idx = idx - 1

        observations = traj_data['obs'][:traj_limit_idx]
        actions = traj_data['actions'][:traj_limit_idx]

        # obs, actions: shape (N * L, ) + S
        # where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # S = (1, ) for discrete space
        # Flatten to (N * L, prod(S))
        if len(observations.shape) > 2:
            #observations = np.reshape(observations, [-1, np.prod(observations.shape[1:])])
            pass
        if len(actions.shape) > 2:
            #actions = np.reshape(actions, [-1, np.prod(actions.shape[1:])])
            pass

        indices = np.random.permutation(len(observations)).astype(np.int64)

        # Train/Validation split when using behavior cloning
        train_indices = indices[:int(train_fraction * len(indices))]
        val_indices = indices[int(train_fraction * len(indices)):]

        assert len(train_indices) > 0, "No sample for the training set"
        assert len(val_indices) > 0, "No sample for the validation set"

        self.observations = observations
        self.actions = actions

        self.returns = traj_data['episode_returns'][:traj_limit_idx]
        self.avg_ret = sum(self.returns) / len(self.returns)
        self.std_ret = np.std(np.array(self.returns))
        self.verbose = verbose

        assert len(self.observations) == len(self.actions), "The number of actions and observations differ " \
                                                            "please check your expert dataset"
        self.num_traj = min(traj_limitation, np.sum(episode_starts))
        self.num_transition = len(self.observations)
        self.randomize = randomize
        self.sequential_preprocessing = sequential_preprocessing

        self.dataloader = None
        self.train_loader = DataLoader(train_indices, self.observations, self.actions, batch_size,
                                       shuffle=self.randomize, start_process=False,
                                       sequential=sequential_preprocessing)
        self.val_loader = DataLoader(val_indices, self.observations, self.actions, batch_size,
                                     shuffle=self.randomize, start_process=False,
                                     sequential=sequential_preprocessing)

        if self.verbose >= 1:
            self.log_info()

    def init_dataloader(self, batch_size):
        """
        Initialize the dataloader used by GAIL.

        :param batch_size: (int)
        """
        indices = np.random.permutation(len(self.observations)).astype(np.int64)
        self.dataloader = DataLoader(indices, self.observations, self.actions, batch_size,
                                     shuffle=self.randomize, start_process=False,
                                     sequential=self.sequential_preprocessing)

    def __del__(self):
        # Exit processes if needed
        for key in self.EXCLUDED_KEYS:
            if self.__dict__.get(key) is not None:
                del self.__dict__[key]

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes processes that are not pickleable
        """
        # Remove processes in order to pickle the dataset.
        return {key: val for key, val in self.__dict__.items() if key not in self.EXCLUDED_KEYS}

    def __setstate__(self, state):
        """
        Restores pickled state.

        init_dataloader() must be called
        after unpickling before using it with GAIL.

        :param state: (dict)
        """
        self.__dict__.update(state)
        for excluded_key in self.EXCLUDED_KEYS:
            assert excluded_key not in state
        self.dataloader = None
        self.train_loader = None
        self.val_loader = None

    def log_info(self):
        """
        Log the information of the dataset.
        """
        logger.log("Total trajectories: {}".format(self.num_traj))
        logger.log("Total transitions: {}".format(self.num_transition))
        logger.log("Average returns: {}".format(self.avg_ret))
        logger.log("Std for returns: {}".format(self.std_ret))

    def get_next_batch(self, split=None):
        """
        Get the batch from the dataset.

        :param split: (str) the type of data split (can be None, 'train', 'val')
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        dataloader = {
            None: self.dataloader,
            'train': self.train_loader,
            'val': self.val_loader
        }[split]

        if dataloader.process is None:
            dataloader.start_process()
        try:
            return next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            return next(dataloader)

    def plot(self):
        """
        Show histogram plotting of the episode returns
        """
        # Isolate dependency since it is only used for plotting and also since
        # different matplotlib backends have further dependencies themselves.
        import matplotlib.pyplot as plt
        plt.hist(self.returns)
        plt.show()


class DataLoader(object):
    """
    A custom dataloader to preprocessing observations (including images)
    and feed them to the network.

    Original code for the dataloader from https://github.com/araffin/robotics-rl-srl
    (MIT licence)
    Authors: Antonin Raffin, René Traoré, Ashley Hill

    :param indices: ([int]) list of observations indices
    :param observations: (np.ndarray) observations or images path
    :param actions: (np.ndarray) actions
    :param batch_size: (int) Number of samples per minibatch
    :param n_workers: (int) number of preprocessing worker (for loading the images)
    :param infinite_loop: (bool) whether to have an iterator that can be reset
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param shuffle: (bool) Shuffle the minibatch after each epoch
    :param start_process: (bool) Start the preprocessing process (default: True)
    :param backend: (str) joblib backend (one of 'multiprocessing', 'sequential', 'threading'
        or 'loky' in newest versions)
    :param sequential: (bool) Do not use subprocess to preprocess the data
        (slower but use less memory for the CI)
    :param partial_minibatch: (bool) Allow partial minibatches (minibatches with a number of element
        lesser than the batch_size)
    """

    def __init__(self, indices, observations, actions, batch_size, n_workers=1,
                 infinite_loop=True, max_queue_len=1, shuffle=False,
                 start_process=True, backend='threading', sequential=False, partial_minibatch=True):
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.indices = indices
        self.original_indices = indices.copy()
        self.n_minibatches = len(indices) // batch_size
        # Add a partial minibatch, for instance
        # when there is not enough samples
        if partial_minibatch and len(indices) % batch_size > 0:
            self.n_minibatches += 1
        self.batch_size = batch_size
        self.observations = observations
        self.actions = actions
        self.shuffle = shuffle
        self.queue = Queue(max_queue_len)
        self.process = None
        self.load_images = isinstance(observations[0], str)
        self.backend = backend
        self.sequential = sequential
        self.start_idx = 0
        if start_process:
            self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        # Skip if in sequential mode
        if self.sequential:
            return
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    @property
    def _minibatch_indices(self):
        """
        Current minibatch indices given the current pointer
        (start_idx) and the minibatch size
        :return: (np.ndarray) 1D array of indices
        """
        return self.indices[self.start_idx:self.start_idx + self.batch_size]

    def sequential_next(self):
        """
        Sequential version of the pre-processing.
        """
        if self.start_idx > len(self.indices):
            raise StopIteration

        if self.start_idx == 0:
            if self.shuffle:
                # Shuffle indices
                np.random.shuffle(self.indices)

        obs = self.observations[self._minibatch_indices]
        if self.load_images:
            obs = np.concatenate([self._make_batch_element(image_path) for image_path in obs],
                                 axis=0)

        actions = self.actions[self._minibatch_indices]
        self.start_idx += self.batch_size
        return obs, actions

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend=self.backend) as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    np.random.shuffle(self.indices)

                for minibatch_idx in range(self.n_minibatches):

                    self.start_idx = minibatch_idx * self.batch_size

                    obs = self.observations[self._minibatch_indices]
                    if self.load_images:
                        if self.n_workers <= 1:
                            obs = [self._make_batch_element(image_path)
                                   for image_path in obs]

                        else:
                            obs = parallel(delayed(self._make_batch_element)(image_path)
                                           for image_path in obs)

                        obs = np.concatenate(obs, axis=0)

                    actions = self.actions[self._minibatch_indices]

                    self.queue.put((obs, actions))

                    # Free memory
                    del obs

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, image_path):
        """
        Process one element.

        :param image_path: (str) path to an image
        :return: (np.ndarray)
        """
        # cv2.IMREAD_UNCHANGED is needed to load
        # grey and RGBa images
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Grey image
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if image is None:
            raise ValueError("Tried to load {}, but it was not found".format(image_path))
        # Convert from BGR to RGB
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((1,) + image.shape)
        return image

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        self.start_idx = 0
        self.indices = self.original_indices.copy()
        return self

    def __next__(self):
        if self.sequential:
            return self.sequential_next()

        if self.process is None:
            raise ValueError("You must call .start_process() before using the dataloader")
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()

# Pre-Train a Model using Behavior Cloning
#import ExpertDataset
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


