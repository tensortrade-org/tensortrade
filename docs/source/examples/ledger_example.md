# Ledger Example

## Install master branch of TensorTrade


```python
!pip install git+https://github.com/tensortrade-org/tensortrade.git -U
```


```python
import tensortrade.env.tt as tt

from tensortrade.feed.core import Stream, DataFeed
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.exchanges.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
```

## Load Data for Exchanges

Using the `tensortrade.data.cdd` module you can load data from any `csv` file provided at:
- https://www.cryptodatadownload.com/data/northamerican/

using the `CryptoDataDownload` class.


```python
cdd = CryptoDataDownload()
coinbase_btc = cdd.fetch("Coinbase", "USD", "BTC", "1h")
coinbase_eth = cdd.fetch("Coinbase", "USD", "ETH", "1h")

bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")
```

## Inspect Transactions of `SimpleOrders`


```python
coinbase = Exchange("coinbase", service=execute_order)(
    Stream.source(list(coinbase_btc['close'][-100:]), dtype="float").rename("USD-BTC"),
    Stream.source(list(coinbase_eth['close'][-100:]), dtype="float").rename("USD-ETH")
)

bitstamp = Exchange("bitstamp", service=execute_order)(
    Stream.source(list(bitstamp_btc['close'][-100:]), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitstamp_eth['close'][-100:]), dtype="float").rename("USD-ETH"),
    Stream.source(list(bitstamp_ltc['close'][-100:]), dtype="float").rename("USD-LTC")
)

portfolio = Portfolio(USD, [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 10 * BTC),
    Wallet(coinbase, 5 * ETH),
    Wallet(bitstamp, 1000 * USD),
    Wallet(bitstamp, 5 * BTC),
    Wallet(bitstamp, 20 * ETH),
    Wallet(bitstamp, 3 * LTC)
])

feed = DataFeed([
    Stream.source(list(bitstamp_eth['volume'][-100:]), dtype="float").rename("volume:/USD-ETH"),
    Stream.source(list(bitstamp_ltc['volume'][-100:]), dtype="float").rename("volume:/USD-LTC")
])

env = tt.create(
    portfolio=portfolio,
    action_scheme=tt.actions.SimpleOrders(),
    reward_scheme=tt.rewards.SimpleProfit(),
    feed=feed
)

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

portfolio.ledger.as_frame().head(7)
```


## Inspect Transactions of `ManagedRiskOrders`


```python
portfolio = Portfolio(USD, [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 0 * BTC),
    Wallet(coinbase, 0 * ETH),
])

env = tt.create(
    portfolio=portfolio,
    action_scheme=tt.actions.ManagedRiskOrders(),
    reward_scheme=tt.rewards.SimpleProfit(),
    feed=feed
)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

portfolio.ledger.as_frame().head(20)
```


## Transactions in Spreadsheets

To take a closer look at the transactions that are happening within the system, copy the transactions to a csv file and load it into any spreadsheet software. If where you are running this allows access to the system clipboard, you can directly copy the frame to your system clipboard by doing the following:
- `portfolio.ledger.as_frame().to_clipboard(index=False)`

Then just paste into any spreadsheet software (e.g. Execel, Google Sheets).
