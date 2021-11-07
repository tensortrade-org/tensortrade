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
START_DATE = '2021-02-09'  # TODO: replace this with your own start date
END_DATE = '2021-09-22'  # TODO: replace this with your own end date

yf_ticker = yfinance.Ticker(ticker=TICKER)

df_training = yf_ticker.history(start=START_DATE, end=END_DATE, interval='60m')
df_training.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
df_training["volume"] = df_training["volume"].astype(int)
df_training.ta.log_return(append=True, length=16)
df_training.ta.rsi(append=True, length=14)
df_training.ta.macd(append=True, fast=12, slow=26)
df_training.to_csv('training.csv', index=False)

df_evaluation = yf_ticker.history(start=START_DATE, end=END_DATE, interval='60m')
df_evaluation.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
df_evaluation["volume"] = df_evaluation["volume"].astype(int)
df_evaluation.ta.log_return(append=True, length=16)
df_evaluation.ta.rsi(append=True, length=14)
df_evaluation.ta.macd(append=True, fast=12, slow=26)
df_evaluation.to_csv('evaluation.csv', index=False)
```
We should now have the two preprocessed files ready (`training.csv` and `evaluation.csv`) 
> Please note that there are many better ways to do this. For example there is an obvious issue in doing things this way: all TA values that we have added require a minimum of 12 samples to be processed, since they perform calculations on longer timeframes. This means that the first 12 rows will not have any meaningful TA values to be trained (or evaluated) with.
> 
> This additional care in retrieving and preprocessing data is left to the user to implement, since many different approaches can be taken, each one with its pros and cons



## Training\evaluation code
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
    dataset = pd.read_csv(filepath_or_buffer=config["csv_filename"], parse_dates=['datetime']).fillna(method='backfill').fillna(method='ffill')
    ttse_commission = 0.0035  # TODO: adjust according to your commission percentage, if present
    price = Stream.source(list(dataset["close"]), dtype="float").rename("USD-TTRD")
    ttse_options = ExchangeOptions(commission=ttse_commission)
    ttse_exchange = Exchange("TTSE", service=execute_order, options=ttse_options)(price)

 # Instruments, Wallets and Portfolio
    USD = Instrument("USD", 2, "US Dollar")
    TTRD = Instrument("TTRD", 2, "TensorTrade Corp")
    cash = Wallet(ttse_exchange, config["start_balance"] * USD)
    asset = Wallet(ttse_exchange, 0 * TTRD)
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

    reward_scheme = default.rewards.SimpleProfit(window_size=7)  # Arbitrarily set as 7
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