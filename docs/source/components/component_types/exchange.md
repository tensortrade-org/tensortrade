## Exchange

Exchanges determine the universe of tradable instruments within a trading environment, return observations to the environment on each time step, and execute trades made within the environment. There are two types of exchanges: live and simulated.

Live exchanges are implementations of `Exchange` backed by live pricing data and a live trade execution engine. For example, `CCXTExchange` is a live exchange, which is capable of returning pricing data and executing trades on hundreds of live cryptocurrency exchanges, such as Binance and Coinbase.

```python
import ccxt
from tensortrade.exchanges.live import CCXTExchange

coinbase = ccxt.coinbasepro()
exchange = CCXTExchange(exchange=coinbase, base_instrument='USD')
```

_There are also exchanges for stock and ETF trading, such as RobinhoodExchange and InteractiveBrokersExchange, but these are still works in progress._

Simulated exchanges, on the other hand, are implementations of `Exchange` backed by simulated pricing data and trade execution.

For example, `FBMExchange` is a simulated exchange, which generates pricing and volume data using fractional brownian motion (FBM). Since its price is simulated, the trades it executes must be simulated as well. The exchange uses a simple slippage model to simulate price and volume slippage on trades, though like almost everything in TensorTrade, this slippage model can easily be swapped out for something more complex.

```python
from tensortrade.exchanges.simulated import FBMExchange

exchange = FBMExchange(base_instrument='BTC', timeframe='1h')
```

Though the `FBMExchange` generates fake price and volume data using a stochastic model, it is simply an implementation of `SimulatedExchange`. Under the hood, `SimulatedExchange` only requires a `data_frame` of price history to generate its simulations. This `data_frame` can either be provided by a coded implementation such as `FBMExchange`, or at runtime.

```python
import pandas as pd
from tensortrade.exchanges.simulated import SimulatedExchange

df = pd.read_csv('./data/btc_ohclv_1h.csv')
exchange = SimulatedExchange(data_frame=df, base_instrument='USD')
```
