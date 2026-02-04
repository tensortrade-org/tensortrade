# Observers and Data Feeds

Observers create what the agent sees. Data feeds process raw data into features. This is where feature engineering happens.

## Learning Objectives

After this tutorial, you will understand:
- How DataFeeds work
- How Observers create observations
- Feature engineering best practices
- Scale-invariant features (critical for generalization)

---

## The Data Pipeline

```
Raw Data ──────> DataFeed ──────> Observer ──────> Agent
(OHLCV)        (streams)       (window)         (sees obs)
```

---

## DataFeed: The Reactive Stream System

DataFeeds use streams to process data reactively.

### Basic Streams

```python
from tensortrade.feed.core import DataFeed, Stream

# Create stream from list
prices = [100, 101, 99, 102, 98]
price_stream = Stream.source(prices, dtype="float").rename("close")

# Create feed
feed = DataFeed([price_stream])
feed.compile()

# Get values one at a time
print(feed.next())  # {"close": 100}
print(feed.next())  # {"close": 101}
print(feed.next())  # {"close": 99}
```

### Stream Operations

```python
price = Stream.source(prices, dtype="float").rename("close")

# Difference from previous value
diff = price.diff().rename("price_change")
# [NaN, 1, -2, 3, -4]

# Percentage change
pct = price.pct_change().rename("returns")
# [NaN, 0.01, -0.0198, 0.0303, -0.0392]

# Rolling operations
ma = price.rolling(window=3).mean().rename("sma_3")
# [NaN, NaN, 100, 100.67, 99.67]

# Apply custom function
def compute_rsi(series, period=14):
    # RSI calculation
    ...

rsi = price.apply(compute_rsi).rename("rsi")
```

### Combining Streams

```python
# Multiple features
close = Stream.source(list(data["close"]), dtype="float").rename("close")
volume = Stream.source(list(data["volume"]), dtype="float").rename("volume")

# Derived features
returns = close.pct_change().rename("returns")
sma_20 = close.rolling(20).mean().rename("sma_20")
trend = ((close - sma_20) / sma_20).rename("trend")

# Combine into feed
feed = DataFeed([close, volume, returns, sma_20, trend])
feed.compile()

# Each feed.next() returns all features
# {"close": 100, "volume": 1234, "returns": 0.01, "sma_20": 99.5, "trend": 0.005}
```

---

## Observer: Creating Observations

The Observer takes DataFeed output and shapes it for the agent.

### TensorTradeObserver

```python
from tensortrade.env.default.observers import TensorTradeObserver

observer = TensorTradeObserver(
    portfolio=portfolio,
    feed=feed,
    renderer_feed=None,
    window_size=10  # Agent sees last 10 steps
)

# Observation shape: (window_size, num_features)
# If 5 features and window_size=10: shape (10, 5)
```

### What the Agent Sees

```
Window of observations (shape: 10 × 5):

Step      close    volume   returns   sma_20   trend
─────────────────────────────────────────────────────
t-9      99,500   1,200    -0.005    99,200   0.003
t-8      99,700   1,100     0.002    99,250   0.005
t-7      99,600   1,300    -0.001    99,300   0.003
t-6      99,900   1,400     0.003    99,350   0.006
t-5     100,000   1,250     0.001    99,400   0.006
t-4      99,800   1,150    -0.002    99,450   0.004
t-3     100,100   1,350     0.003    99,500   0.006
t-2     100,200   1,200     0.001    99,550   0.007
t-1     100,000   1,100    -0.002    99,600   0.004
t       100,300   1,400     0.003    99,650   0.007  ← current
```

The agent learns patterns across this window.

---

## Feature Engineering

### The Problem with Raw Prices

```python
# BAD: Raw prices
features = [
    price,  # $100,000 today
    sma_50  # $98,000
]

# Training data: BTC at $50,000
# Test data: BTC at $100,000
# Agent learns: "price > $80,000 = expensive, sell"
# This doesn't generalize!
```

### Scale-Invariant Features

Features that don't depend on absolute price level:

```python
# GOOD: Scale-invariant features
features = [
    returns_1h,              # -0.5% to +0.5% regardless of price
    (price - sma_50) / sma_50,  # Relative deviation from average
    rsi / 100,               # Already bounded 0-1
    volume / volume_sma_20,  # Ratio, not absolute
]

# These work at ANY price level!
```

### Best Feature Set (from experiments)

```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-invariant features."""
    df = df.copy()

    # Returns (different timeframes)
    for p in [1, 4, 12, 24, 48]:
        df[f'ret_{p}h'] = np.tanh(df['close'].pct_change(p) * 10)

    # RSI (normalized to [-1, 1])
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50

    # Trend (relative to SMAs)
    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    df['trend_20'] = np.tanh((df['close'] - sma20) / sma20 * 10)
    df['trend_50'] = np.tanh((df['close'] - sma50) / sma50 * 10)
    df['trend_strength'] = np.tanh((sma20 - sma50) / sma50 * 20)

    # Volatility (normalized)
    df['vol'] = df['close'].rolling(24).std() / df['close']
    df['vol_norm'] = np.tanh((df['vol'] - df['vol'].rolling(72).mean())
                              / df['vol'].rolling(72).std())

    # Volume (ratio)
    df['vol_ratio'] = np.log1p(df['volume'] / df['volume'].rolling(20).mean())

    # Bollinger Band position (bounded 0-1)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_pos'] = ((df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std)).clip(0, 1)

    return df.bfill().ffill()
```

### Why Each Feature

| Feature | Purpose | Range |
|---------|---------|-------|
| `ret_1h, ret_4h, ...` | Momentum at different timeframes | [-1, 1] (tanh) |
| `rsi` | Overbought/oversold | [-1, 1] |
| `trend_20, trend_50` | Price vs moving averages | [-1, 1] (tanh) |
| `trend_strength` | Trend direction | [-1, 1] (tanh) |
| `vol_norm` | Relative volatility | [-1, 1] (tanh) |
| `vol_ratio` | Volume anomaly | ~[-1, 2] (log) |
| `bb_pos` | Bollinger band position | [0, 1] |

---

## Normalization Techniques

### tanh Scaling

```python
# Bounds unbounded values to [-1, 1]
feature = np.tanh(raw_value * scale_factor)

# Example: 5% return → tanh(0.05 * 10) = tanh(0.5) ≈ 0.46
# Example: -10% return → tanh(-0.10 * 10) = tanh(-1) ≈ -0.76
```

### Z-Score

```python
# Normalizes to mean=0, std=1 (approximately)
feature = (value - rolling_mean) / rolling_std

# Then optionally bound with tanh
feature_bounded = np.tanh(feature)
```

### Min-Max to [0, 1]

```python
# Good for bounded indicators like RSI
feature = (value - min_val) / (max_val - min_val)

# RSI is naturally 0-100
rsi_normalized = rsi / 100  # Now 0-1
```

### Log Transform

```python
# Good for skewed distributions (like volume)
feature = np.log1p(value / baseline)

# np.log1p(x) = ln(1 + x), handles x=0 gracefully
```

---

## Creating a Feed from DataFrame

```python
import pandas as pd
from tensortrade.feed.core import DataFeed, Stream

# Your data
data = pd.DataFrame({
    'close': [100, 101, 99, 102],
    'volume': [1000, 1100, 900, 1200],
})

# Add features
data = add_features(data)

# Select feature columns
feature_cols = ['ret_1h', 'rsi', 'trend_20', 'vol_norm']

# Create streams
streams = []
for col in feature_cols:
    stream = Stream.source(list(data[col]), dtype="float").rename(col)
    streams.append(stream)

# Create feed
feed = DataFeed(streams)
feed.compile()
```

---

## Observer Configuration

### Window Size

```python
observer = TensorTradeObserver(
    feed=feed,
    window_size=10  # How much history agent sees
)
```

**Trade-offs:**
- Larger window = more context, but harder to learn patterns
- Smaller window = faster learning, but less history
- Typical range: 10-30 steps

**From experiments:** `window_size=17` performed well.

### Observation Space

```python
# Observer automatically creates observation_space
observer.observation_space
# Box(shape=(window_size, num_features), dtype=float32)

# Example: window=10, features=13
# shape = (10, 13)
```

---

## Common Mistakes

### Mistake 1: Using Raw Prices

```python
# BAD
feed = DataFeed([
    Stream.source(list(data['close']), dtype="float").rename("close"),
])

# Model learns "$100,000 is high" - doesn't generalize
```

**Fix:** Use returns or relative deviations instead.

### Mistake 2: Not Handling NaN

```python
# BAD: Rolling operations produce NaN at start
sma = data['close'].rolling(20).mean()
# First 19 values are NaN!

# If fed to model, can cause errors or NaN gradients
```

**Fix:** Use `.bfill().ffill()` or drop NaN rows:

```python
data = add_features(data).bfill().ffill()
# Or
data = add_features(data).dropna()
```

### Mistake 3: Look-Ahead Bias

```python
# BAD: Using future data
data['tomorrow_return'] = data['close'].pct_change().shift(-1)  # FUTURE!

# Model sees tomorrow's return to predict today
# Training looks amazing, real trading fails
```

**Fix:** Only use `.shift(positive_value)` or no shift for current values.

### Mistake 4: Too Many Features

```python
# BAD: 50+ features
features = ['open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
            'rsi_7', 'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            ...]  # 50 features

# Model memorizes patterns in training data (overfitting)
```

**From experiments:**
```
34 features: Test P&L -$2,690 (severe overfitting)
13 features: Test P&L -$650 (much better generalization)
```

---

## Putting It Together

Complete example of creating feed and observer:

```python
import pandas as pd
import numpy as np
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.env.default.observers import TensorTradeObserver
from tensortrade.data.cdd import CryptoDataDownload

# 1. Get data
cdd = CryptoDataDownload()
data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

# 2. Add features
def add_features(df):
    df = df.copy()
    # Returns
    for p in [1, 4, 12, 24]:
        df[f'ret_{p}h'] = np.tanh(df['close'].pct_change(p) * 10)
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50
    # Trend
    sma20 = df['close'].rolling(20).mean()
    df['trend'] = np.tanh((df['close'] - sma20) / sma20 * 10)
    return df.bfill().ffill()

data = add_features(data)

# 3. Create streams
feature_cols = ['ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'rsi', 'trend']
streams = [Stream.source(list(data[c]), dtype="float").rename(c)
           for c in feature_cols]

# 4. Create feed
feed = DataFeed(streams)
feed.compile()

# 5. Create observer (with portfolio from your setup)
observer = TensorTradeObserver(
    portfolio=portfolio,
    feed=feed,
    window_size=10
)

# Observer is now ready to provide observations to agent
```

---

## Key Takeaways

1. **DataFeed uses streams** for reactive data processing
2. **Observers create windowed observations** from feed output
3. **Scale-invariant features are critical** - use returns, ratios, normalized values
4. **Fewer features = better generalization** - 5-13 features is often optimal
5. **Always handle NaN** from rolling operations
6. **Never use future data** - only past and current

---

## Checkpoint

Before continuing, verify you understand:

- [ ] Why raw prices don't generalize (price level dependence)
- [ ] What tanh scaling does and why it's useful
- [ ] What window_size means for observations
- [ ] How to avoid look-ahead bias in features

---

## Next Steps

You've learned the core components. Now train a real agent:
- [First Training](../04-training/01-first-training.md) - Train with Ray RLlib
