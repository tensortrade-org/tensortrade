# Renderers and Plotly Chart

<br>**Data Loading Function**<br>


```bash
# ipywidgets is required to run Plotly in Jupyter Notebook.
# Uncomment and run the following line to install it if required.

!pip install ipywidgets
```


```python
import ta

import pandas as pd

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio

%matplotlib inline
```


```python
def load_csv(filename):
    df = pd.read_csv('data/' + filename, skiprows=1)
    df.drop(columns=['symbol', 'volume_btc'], inplace=True)

    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    df['date'] = df['date'].str[:14] + '00-00 ' + df['date'].str[-2:]

    # Convert the date column type from string to datetime for proper sorting.
    df['date'] = pd.to_datetime(df['date'])

    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by='date', ascending=True, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')

    return df
```


```python
df = load_csv('Bitfinex_BTCUSD_1h.csv')
df.head()
```

<br>**Data Preparation**<br>


```python
dataset = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna=True)
dataset.head(3)
```

Note: It is recommended to create the chart data *after* creating and cleaning the dataset to ensure one-to-one mapping between the historical prices data and the dataset.


```python
price_history = dataset[['date', 'open', 'high', 'low', 'close', 'volume']]  # chart data
display(price_history.head(3))

dataset.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)
```


<br>**Setup Trading Environment**<br>


```python
bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-BTC")
)

portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 10 * BTC),
])

with NameSpace("bitfinex"):
    streams = [Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns]

feed = DataFeed(streams)
feed.next()
```




    {'bitfinex:/volume_adi': 503274.35945218964,
     'bitfinex:/volume_obv': 0.0,
     'bitfinex:/volume_cmf': 0.5388828039430464,
     'bitfinex:/volume_fi': 0.0,
     'bitfinex:/volume_em': 0.0,
     'bitfinex:/volume_vpt': -187039.68188942864,
     'bitfinex:/volume_nvi': 1000.0,
     'bitfinex:/volatility_atr': 88.87448632521046,
     'bitfinex:/volatility_bbh': 2509.17,
     'bitfinex:/volatility_bbl': 2509.17,
     'bitfinex:/volatility_bbm': 2509.17,
     'bitfinex:/volatility_bbhi': 0.0,
     'bitfinex:/volatility_bbli': 0.0,
     'bitfinex:/volatility_kcc': 2505.89,
     'bitfinex:/volatility_kch': 2524.15,
     'bitfinex:/volatility_kcl': 2487.6299999999997,
     'bitfinex:/volatility_kchi': 0.0,
     'bitfinex:/volatility_kcli': 0.0,
     'bitfinex:/volatility_dch': 2509.17,
     'bitfinex:/volatility_dcl': 2509.17,
     'bitfinex:/volatility_dchi': 0.0,
     'bitfinex:/volatility_dcli': 0.0,
     'bitfinex:/trend_macd': 0.0,
     'bitfinex:/trend_macd_signal': 0.0,
     'bitfinex:/trend_macd_diff': 0.0,
     'bitfinex:/trend_ema_fast': 2509.17,
     'bitfinex:/trend_ema_slow': 2509.17,
     'bitfinex:/trend_adx': 0.0,
     'bitfinex:/trend_adx_pos': 0.0,
     'bitfinex:/trend_adx_neg': 0.0,
     'bitfinex:/trend_vortex_ind_pos': 1.0,
     'bitfinex:/trend_vortex_ind_neg': 1.0,
     'bitfinex:/trend_vortex_diff': 0.0,
     'bitfinex:/trend_trix': -65.01942947444225,
     'bitfinex:/trend_mass_index': 1.0,
     'bitfinex:/trend_cci': 0.0,
     'bitfinex:/trend_dpo': 4669.658895132072,
     'bitfinex:/trend_kst': -650.476416605854,
     'bitfinex:/trend_kst_sig': -650.476416605854,
     'bitfinex:/trend_kst_diff': 0.0,
     'bitfinex:/trend_ichimoku_a': 2504.25,
     'bitfinex:/trend_ichimoku_b': 2504.25,
     'bitfinex:/trend_visual_ichimoku_a': 7164.427851548871,
     'bitfinex:/trend_visual_ichimoku_b': 7151.343258415852,
     'bitfinex:/trend_aroon_up': 4.0,
     'bitfinex:/trend_aroon_down': 4.0,
     'bitfinex:/trend_aroon_ind': 0.0,
     'bitfinex:/momentum_rsi': 50.0,
     'bitfinex:/momentum_mfi': 50.0,
     'bitfinex:/momentum_tsi': -100.0,
     'bitfinex:/momentum_uo': 0.29997594458961346,
     'bitfinex:/momentum_stoch': 76.94414019715232,
     'bitfinex:/momentum_stoch_signal': 76.94414019715232,
     'bitfinex:/momentum_wr': -23.055859802847678,
     'bitfinex:/momentum_ao': 0.0,
     'bitfinex:/momentum_kama': 2509.17,
     'bitfinex:/others_dr': -65.0476416605854,
     'bitfinex:/others_dlr': 0.0,
     'bitfinex:/others_cr': 0.0}



<br>**Trading Environment Renderers**<br>
A renderer is a channel for the trading environment to output its current state. One or more renderers can be attached to the environment at the same time. For example, you can let the environment draw a chart and log to a file at the same time.

Notice that while all renderers can technically be used together, you need to select the best combination to avoid undesired results. For example, PlotlyTradingChart can work well with FileLogger but may not display well with ScreenLogger.

Renderer can be set by name (string) or class, single or list. Available renderers are:
* `'screenlog'` or `ScreenLogger`: Shows results on the screen.
* `'filelog'` or `FileLogger`: Logs results to a file.
* `'plotly'` or `PlotlyTradingChart`: A trading chart based on Plotly.

<br>**Examples**:<br>

* renderers = 'screenlog' (default)
* renderers = ['screenlog', 'filelog']
* renderers = ScreenLogger()
* renderers = ['screenlog', `FileLogger()`]
* renderers = [`FileLogger(filename='example.log')`]

Renderers can also be created and configured first then attached to the environment as seen in a following example.


```python
import tensortrade.env.default as default

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer="screen-log",  # ScreenLogger used with default settings
    window_size=20
)
```


```python
from tensortrade.agents import DQNAgent

agent = DQNAgent(env)
agent.train(n_episodes=2, n_steps=200, render_interval=10)
```

    ====      AGENT ID: b8b6ad1a-c158-4743-8a2d-aab3a26ce82c      ====
    [2020-07-29 3:43:06 PM] Episode: 1/2 Step: 131/200
    [2020-07-29 3:43:07 PM] Episode: 1/2 Step: 141/200
    [2020-07-29 3:43:08 PM] Episode: 1/2 Step: 151/200
    [2020-07-29 3:43:09 PM] Episode: 1/2 Step: 161/200
    [2020-07-29 3:43:11 PM] Episode: 1/2 Step: 171/200
    [2020-07-29 3:43:12 PM] Episode: 1/2 Step: 181/200
    [2020-07-29 3:43:13 PM] Episode: 1/2 Step: 191/200
    [2020-07-29 3:43:14 PM] Episode: 1/2 Step: 201/200
    [2020-07-29 3:43:15 PM] Episode: 2/2 Step: 11/200
    [2020-07-29 3:43:16 PM] Episode: 2/2 Step: 21/200
    [2020-07-29 3:43:17 PM] Episode: 2/2 Step: 31/200
    [2020-07-29 3:43:19 PM] Episode: 2/2 Step: 41/200
    [2020-07-29 3:43:20 PM] Episode: 2/2 Step: 51/200
    [2020-07-29 3:43:21 PM] Episode: 2/2 Step: 61/200
    [2020-07-29 3:43:22 PM] Episode: 2/2 Step: 71/200
    [2020-07-29 3:43:23 PM] Episode: 2/2 Step: 81/200
    [2020-07-29 3:43:24 PM] Episode: 2/2 Step: 91/200
    [2020-07-29 3:43:25 PM] Episode: 2/2 Step: 101/200
    [2020-07-29 3:43:26 PM] Episode: 2/2 Step: 111/200
    [2020-07-29 3:43:27 PM] Episode: 2/2 Step: 121/200
    [2020-07-29 3:43:29 PM] Episode: 2/2 Step: 131/200
    [2020-07-29 3:43:30 PM] Episode: 2/2 Step: 141/200
    [2020-07-29 3:43:31 PM] Episode: 2/2 Step: 151/200
    [2020-07-29 3:43:32 PM] Episode: 2/2 Step: 161/200
    [2020-07-29 3:43:33 PM] Episode: 2/2 Step: 171/200
    [2020-07-29 3:43:34 PM] Episode: 2/2 Step: 181/200
    [2020-07-29 3:43:35 PM] Episode: 2/2 Step: 191/200
    [2020-07-29 3:43:36 PM] Episode: 2/2 Step: 201/200





    -125697.1219732128



<br>**Environment with Multiple Renderers**<br>
Create PlotlyTradingChart and FileLogger renderers. Configuring renderers is optional as they can be used with their default settings.


```python
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger

chart_renderer = PlotlyTradingChart(
    display=True,  # show the chart on screen (default)
    height=800,  # affects both displayed and saved file height. None for 100% height.
    save_format="html",  # save the chart to an HTML file
    auto_open_html=True,  # open the saved HTML chart in a new browser tab
)

file_logger = FileLogger(
    filename="example.log",  # omit or None for automatic file name
    path="training_logs"  # create a new directory if doesn't exist, None for no directory
)
```

<br>**Environement with Multiple Renderers**<br>

With the plotly renderer you must provide an parameter called `renderer_feed`. This is a `DataFeed` instance that provides all the information that is required by a renderer to function.


```python
renderer_feed = DataFeed([
    Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
)

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    window_size=20,
    renderer_feed=renderer_feed,
    renderers=[
        chart_renderer,
        file_logger
    ]
)
```

<br>**Setup and Train DQN Agent**<br>
The green and red arrows shown on the chart represent buy and sell trades respectively. The head of each arrow falls at the trade execution price.


```python
from tensortrade.agents import DQNAgent

agent = DQNAgent(env)

# Set render_interval to None to render at episode ends only
agent.train(n_episodes=2, n_steps=200, render_interval=10)
```

    ====      AGENT ID: fec5e2c5-eb35-4ff6-8416-b876f0e8be66      ====

![image](https://user-images.githubusercontent.com/4431953/149672722-6305a8d8-7a95-477a-b83a-d81ad4cdad17.png)




    -122271.41943956864



<br>**Direct Performance and Net Worth Plotting**<br>
Alternatively, the final performance and net worth can be displayed using `Pandas` via `Matplotlib`.


```python
%matplotlib inline
import pandas as pd
df = pd.DataFrame(portfolio.performance)
```

Transpose the dataframe using `T` and call `.plot()`

```
df.T.plot()
```


![png](renderers_and_plotly_chart_files/renderers_and_plotly_chart_21_1.png)



```python
df.loc["net_worth"].plot()
```

![png](renderers_and_plotly_chart_files/renderers_and_plotly_chart_22_1.png)
