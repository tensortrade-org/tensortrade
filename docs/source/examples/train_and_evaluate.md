# Train and Evaluate


```python
!python3 -m pip install git+https://github.com/tensortrade-org/tensortrade.git
```

<br>**Setup Data Fetching**<br>


```python
import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent


%matplotlib inline
```


```python
cdd = CryptoDataDownload()

data = cdd.fetch("Coinbase", "USD", "BTC", "1h")
```

<br>**Create features with the feed module**<br>


```python
def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


features = []
for c in data.columns[1:]:
    s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
    features += [s]

cp = Stream.select(features, lambda s: s.name == "close")

features = [
    cp.log().diff().rename("lr"),
    rsi(cp, period=20).rename("rsi"),
    macd(cp, fast=10, slow=50, signal=5).rename("macd")
]

feed = DataFeed(features)
feed.compile()
```


```python
for i in range(5):
    print(feed.next())
```

    {'lr': nan, 'rsi': nan, 'macd': 0.0}
    {'lr': -0.008300031641449657, 'rsi': 0.0, 'macd': -1.9717171717171975}
    {'lr': -0.01375743446296962, 'rsi': 0.0, 'macd': -6.082702245269603}
    {'lr': 0.0020025323250756344, 'rsi': 8.795475693113076, 'macd': -7.287625162566419}
    {'lr': 0.00344213459739251, 'rsi': 21.34663357024277, 'macd': -6.522181201739986}


<br>**Setup Trading Environment**<br>


```python
coinbase = Exchange("coinbase", service=execute_order)(
    Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
)

portfolio = Portfolio(USD, [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 10 * BTC)
])


renderer_feed = DataFeed([
    Stream.source(list(data["date"])).rename("date"),
    Stream.source(list(data["open"]), dtype="float").rename("open"),
    Stream.source(list(data["high"]), dtype="float").rename("high"),
    Stream.source(list(data["low"]), dtype="float").rename("low"),
    Stream.source(list(data["close"]), dtype="float").rename("close"),
    Stream.source(list(data["volume"]), dtype="float").rename("volume")
])


env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer_feed=renderer_feed,
    renderer=default.renderers.PlotlyTradingChart(),
    window_size=20
)
```


```python
env.observer.feed.next()
```




    {'internal': {'coinbase:/USD-BTC': 2509.17,
      'coinbase:/USD:/free': 10000.0,
      'coinbase:/USD:/locked': 0.0,
      'coinbase:/USD:/total': 10000.0,
      'coinbase:/BTC:/free': 10.0,
      'coinbase:/BTC:/locked': 0.0,
      'coinbase:/BTC:/total': 10.0,
      'coinbase:/BTC:/worth': 25091.7,
      'net_worth': 35091.7},
     'external': {'lr': nan, 'rsi': nan, 'macd': 0.0},
     'renderer': {'date': Timestamp('2017-07-01 11:00:00'),
      'open': 2505.56,
      'high': 2513.38,
      'low': 2495.12,
      'close': 2509.17,
      'volume': 287000.32}}



<br>**Setup and Train DQN Agent**<br>


```python
agent = DQNAgent(env)

agent.train(n_steps=200, n_episodes=2, save_path="agents/")
```
