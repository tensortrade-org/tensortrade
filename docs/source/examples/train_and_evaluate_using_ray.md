# Train and Evaluate using Ray
This doc has been updated to leverage Ray as a training backend. Builtin agents have been deprecated, since Ray offers a much more stable and faster execution, on top of the parallelized training\evaluation in a distributed cluster.
> We will be using the fictional TensorTrade Corp stock (**TTRD**) traded in the TensorTrade Stock Exchange (**TTSE**) in all the examples, hoping noone goes public under that ticker and generates some confusion :)
> The base currency will be US Dollars (**USD**) with a commission of 0.35% on each trade done.


## Installing Requirements



### TensorTrade
You can follow the "easy" way and install using the provided PIP package: (latest version available at the time of this example is 1.0.3)
```console
pip install tensortrade
```
... or you can install the latest code available, pulling directly from the development codebase (this is the suggested way, since many fixes take time to get packaged in a new release)  
```console
pip install git+https://github.com/tensortrade-org/tensortrade.git
```



### Ray
This example focuses on having Ray as a training backend, so we need to install `ray` with the `default` base features and `rrlib` and `tune` features (Latest version available at the time of this example is 1.8.0)
```console
pip install ray[default,rllib,tune]==1.8.0
```
Ray has many builtins [RLLib algorithms](https://docs.ray.io/en/latest/rllib-algorithms.html) to train agents with, and together with [Tune](https://docs.ray.io/en/latest/tune/index.html) it allows us to run multiple parallel (or sequential) trainings with different parameters in a grid search for the best one to use within TensorTrade.
> Pay attention to where you are installing Ray. Until recently (1.8.0 does *NOT* have this issue, but previous versions do), if Ray was installed in a VENV, it would hang at `ray.init()` during execution. If you need to be using Ray versions prior to 1.8.0 please make sure you are installing Ray in the default system environment.



### Other Requirements
On top of TensorTrade and Ray, you need to install whichever library you want to use to fetch the data you want to train on, and any library you plan to use to augment your source data with calculated technical analysis fields such as `log return`, `rsi`, `macd` and so on...
In this example I am using `yfinance` to fetch stock ticker data and `pandas-ta` to add technical indicators.
I know pandas-ta is not actively mantained, but it's well enough for what I need, and it's really simple to use (you'll see later). Also, for pandas-ta, you need to specify the `--pre` parameter when installing the package since pandas-ta has not had a stable release yet, and pip only looks for stable releases by default.
```console
pip install yfinance==0.1.64
pip install pandas-ta==0.3.14b --pre
```
Of course you can choose whichever providers\augmenters you want, and this is just an example to show one possible implementation.



## Build the training\evaluation dataset
We now arbitrarily define the timeframes we want to use in training our agent. Please note **you have to substitute the ticker name**, as it appears on Yahoo! Finance (ie: Gold Futures is `GC=F`, CBOE Volatility Index is `^VIX`, GameStop Corp is `GME`, and so on...).

Once retrieved we need to add the relevant TA values, and this is enabled just by importing `pandas_ta` and then accessing the `ta` child in a dataframe. The `#noqa` tag is to silence PyCharm from reporting that `pandas_ta` has been imported but it has not been used (which is not true, but it can't know that)

In the end, to avoid continuous Yahoo! Finance API requests, we simply save those dataframes to CSV, so that we can then simply read those preprocessed files during training and evaluation. 
```python
import yfinance
import pandas_ta  #noqa
TICKER = 'TTRD'  # TODO: replace this with your own ticker
TRAIN_START_DATE = '2021-02-09'  # TODO: replace this with your own start date
TRAIN_END_DATE = '2021-09-30'  # TODO: replace this with your own end date
EVAL_START_DATE = '2021-10-01'  # TODO: replace this with your own end date
EVAL_END_DATE = '2021-11-12'  # TODO: replace this with your own end date

yf_ticker = yfinance.Ticker(ticker=TICKER)

df_training = yf_ticker.history(start=TRAIN_START_DATE, end=TRAIN_END_DATE, interval='60m')
df_training.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
df_training["Volume"] = df_training["Volume"].astype(int)
df_training.ta.log_return(append=True, length=16)
df_training.ta.rsi(append=True, length=14)
df_training.ta.macd(append=True, fast=12, slow=26)
df_training.to_csv('training.csv', index=False)

df_evaluation = yf_ticker.history(start=EVAL_START_DATE, end=EVAL_END_DATE, interval='60m')
df_evaluation.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
df_evaluation["Volume"] = df_evaluation["Volume"].astype(int)
df_evaluation.ta.log_return(append=True, length=16)
df_evaluation.ta.rsi(append=True, length=14)
df_evaluation.ta.macd(append=True, fast=12, slow=26)
df_evaluation.to_csv('evaluation.csv', index=False)
```
We should now have the two preprocessed files ready (`training.csv` and `evaluation.csv`) 
> Please note that there are many better ways to do this. For example there is an obvious issue in doing things this way: all TA values that we have added require a minimum of 12 samples to be processed, since they perform calculations on longer timeframes. This means that the first 12 rows will not have any meaningful TA values to be trained (or evaluated) with.
> 
> This additional care in retrieving and preprocessing data is left to the user to implement, since many different approaches can be taken, each one with its pros and cons



## Training and Evaluation



### Create the environment build function
Here we are using the `config` dictionary to store the CSV filename that we need to read. During the training phase, we will pass `training.csv` as the value, while during the evaluation phase we will pass `evaluation.csv`
```python
import pandas as pd
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default

def create_env(config):
    dataset = pd.read_csv(filepath_or_buffer=config["csv_filename"], parse_dates=['Datetime']).fillna(method='backfill').fillna(method='ffill')
    ttse_commission = 0.0035  # TODO: adjust according to your commission percentage, if present
    price = Stream.source(list(dataset["Close"]), dtype="float").rename("USD-TTRD")
    ttse_options = ExchangeOptions(commission=ttse_commission)
    ttse_exchange = Exchange("TTSE", service=execute_order, options=ttse_options)(price)

 # Instruments, Wallets and Portfolio
    USD = Instrument("USD", 2, "US Dollar")
    TTRD = Instrument("TTRD", 2, "TensorTrade Corp")
    cash = Wallet(ttse_exchange, 1000 * USD)  # This is the starting cash we are going to use
    asset = Wallet(ttse_exchange, 0 * TTRD)  # And we will start owning 0 stocks of TTRD
    portfolio = Portfolio(USD, [cash, asset])

    # Renderer feed
    renderer_feed = DataFeed([
        Stream.source(list(dataset["Datetime"])).rename("date"),
        Stream.source(list(dataset["Open"]), dtype="float").rename("open"),
        Stream.source(list(dataset["High"]), dtype="float").rename("high"),
        Stream.source(list(dataset["Low"]), dtype="float").rename("low"),
        Stream.source(list(dataset["Close"]), dtype="float").rename("close"),
        Stream.source(list(dataset["Volume"]), dtype="float").rename("volume")
    ])

    features = []
    for c in dataset.columns[1:]:
        s = Stream.source(list(dataset[c]), dtype="float").rename(dataset[c].name)
        features += [s]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = default.rewards.SimpleProfit(window_size=config["reward_window_size"])
    action_scheme = default.actions.BSH(cash=cash, asset=asset)
    
    env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer_feed=renderer_feed,
            renderer=[],
            window_size=config["window_size"],
            max_allowed_loss=config["max_allowed_loss"]
        )
    
    return env
```


### Initialize and run Ray
Now it's time to actually initialize and run Ray, passing all the parameters necessary, including the name of the environment creator function (`create_env` defined above).

**Please note that many of these parameters need to be tuned for your specific use case** (ie: `"training_iteration": 5` is ***way*** too few to get anything remotely useful, but it allows the example to quickly reach the end)
```python
import ray
import os
from ray import tune
from ray.tune.registry import register_env

# Let's define some tuning parameters
FC_SIZE = tune.grid_search([[256, 256], [1024], [128, 64, 32]])  # Those are the alternatives that ray.tune will try...
LEARNING_RATE = tune.grid_search([0.001, 0.0005, 0.00001])  # ... and they will be combined with these ones ...
MINIBATCH_SIZE = tune.grid_search([5, 10, 20])  # ... and these ones, in a cartesian product.

# Get the current working directory
cwd = os.getcwd()

# Initialize Ray
ray.init()  # There are *LOTS* of initialization parameters, like specifying the maximum number of CPUs\GPUs to allocate. For now just leave it alone.

# Register our environment, specifying which is the environment creation function
register_env("MyTrainingEnv", create_env)

# Specific configuration keys that will be used during training
env_config_training = {
    "window_size": 14,  # We want to look at the last 14 samples (hours)
    "reward_window_size": 7,  # And calculate reward based on the actions taken in the next 7 hours
    "max_allowed_loss": 0.10,  # If it goes past 10% loss during the iteration, we don't want to waste time on a "loser".
    "csv_filename": os.path.join(cwd, 'training.csv'),  # The variable that will be used to differentiate training and validation datasets
}
# Specific configuration keys that will be used during evaluation (only the overridden ones)
env_config_evaluation = {
    "max_allowed_loss": 1.00,  # During validation runs we want to see how bad it would go. Even up to 100% loss.
    "csv_filename": os.path.join(cwd, 'evaluation.csv'),  # The variable that will be used to differentiate training and validation datasets
}

analysis = tune.run(
    run_or_experiment="PPO",  # We'll be using the builtin PPO agent in RLLib
    name="MyExperiment1",
    metric='episode_reward_mean',
    mode='max',
    stop={
        "training_iteration": 5  # Let's do 5 steps for each hyperparameter combination
    },
    config={
        "env": "MyTrainingEnv",
        "env_config": env_config_training,  # The dictionary we built before
        "log_level": "WARNING",
        "framework": "torch",
        "ignore_worker_failures": True,
        "num_workers": 1,  # One worker per agent. You can increase this but it will run fewer parallel trainings.
        "num_envs_per_worker": 1,
        "num_gpus": 0,  # I yet have to understand if using a GPU is worth it, for our purposes, but I think it's not. This way you can train on a non-gpu enabled system.
        "clip_rewards": True,
        "lr": LEARNING_RATE,  # Hyperparameter grid search defined above
        "gamma": 0.50,  # This can have a big impact on the result and needs to be properly tuned (range is 0 to 1)
        "observation_filter": "MeanStdFilter",
        "model": {
            "fcnet_hiddens": FC_SIZE,  # Hyperparameter grid search defined above
        },
        "sgd_minibatch_size": MINIBATCH_SIZE,  # Hyperparameter grid search defined above
        "evaluation_interval": 1,  # Run evaluation on every iteration
        "evaluation_config": {
            "env_config": env_config_evaluation,  # The dictionary we built before (only the overriding keys to use in evaluation)
            "explore": False,  # We don't want to explore during evaluation. All actions have to be repeatable.
        },
    },
    num_samples=1,  # Have one sample for each hyperparameter combination. You can have more to average out randomness.
    keep_checkpoints_num=10,  # Keep the last 2 checkpoints
    checkpoint_freq=1,  # Do a checkpoint on each iteration (slower but you can pick more finely the checkpoint to use later)
)
```
Once you launch this, it will block (meaning it will stay running until the stop condition happens for all samples). You will receive a console output that will show the training progress and all the hyperparameters that are being used in each trial.


## Monitoring
You can basically monitor two things: how Ray is behaving on your cluster (local or distributed, in this example it will be a local cluster), and how is the training proceeding within the TensorTrade environment.

### Ray Dashboard
The Ray Dashboard can be accessed by default at [http://127.0.0.1:8265](http://127.0.0.1:8265). If you want to access it remotely, you just need to specify `dashboard_host="0.0.0.0"` as a ray.init() parameter. This will allow external\remote connections to the Dashboard, provided the newtork routing\accessibility and eventual firewall is correctly configured.

The Dashboard will show resource usage statistics on the nodes working on the cluster, most importantly CPU and RAM usage. Please refer to the [official dashboard documentation](https://docs.ray.io/en/latest/ray-dashboard.html) for further info on that.

### TensorBoard
In order to browse the TensorBoard you first need to launch it, running this console command:
```console
tensorboard --logdir path\to\Ray\results\folder
```
You can then access it by default at [http://127.0.0.1:6006](http://127.0.0.1:6006). As with the Ray Dashboard, if you want to access it remotely, you need to specify `--host 0.0.0.0` in the commandline parameter, like:
```console
tensorboard --logdir path\to\Ray\results\folder --host 0.0.0.0
```

The most important values you need to watch out for during a training are `tune/episode_reward_min`, `tune/episode_reward_mean` and `tune/episode_reward_max` that represent the minimum, average and maximum reward obtained by the agent during training on that specific iteration, using the training dataset.

Alongside with those three, there are the evaluation counterparts, so `tune/evaluation/episode_reward_min`, `tune/evaluation/episode_reward_mean` and `tune/evaluation/episode_reward_max` which represent the same metric, calculated on the evaluation dataset.

The best model will be the one that will score the highest evaluation values, given that the training values will be always higher\better than the evaluation ones.
