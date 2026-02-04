# TensorTrade Architecture

This tutorial explains how TensorTrade's components work together. Understanding the architecture helps you customize and extend the framework.

## Learning Objectives

After reading this tutorial, you will understand:
- How the TradingEnv episode loop works
- What each component does
- How components communicate
- How to choose the right components for your task

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TradingEnv                                 │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     Episode Loop                             │  │
│   │                                                              │  │
│   │  1. Observer generates observation from market data          │  │
│   │  2. Agent (external) selects action                          │  │
│   │  3. ActionScheme converts action to orders                   │  │
│   │  4. Portfolio executes orders through Exchange               │  │
│   │  5. RewardScheme computes reward                             │  │
│   │  6. Stopper checks if episode should end                     │  │
│   │  7. Loop back to step 1                                      │  │
│   │                                                              │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   Components:                                                       │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│   │ Observer │ │ Action   │ │ Reward   │ │ Stopper  │ │ Renderer ││
│   │          │ │ Scheme   │ │ Scheme   │ │          │ │          ││
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│                                                                     │
│   Supporting:                                                       │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│   │ DataFeed │ │ Exchange │ │ Broker   │ │Portfolio │              │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Episode Loop in Detail

Let's trace one step through the system:

```python
# In your training loop:
obs, info = env.reset()
action = agent.compute_action(obs)
obs, reward, done, truncated, info = env.step(action)
```

What happens inside `env.step(action)`:

```
Step 1: ActionScheme interprets action
┌─────────────────────────────────────────────────┐
│ action = 0 (BUY signal from agent)              │
│                                                 │
│ ActionScheme.perform(action):                   │
│   - Current position: CASH                      │
│   - Action 0 means: switch to BTC               │
│   - Creates order: BUY all USD worth of BTC     │
│   - Submits to Broker                           │
└─────────────────────────────────────────────────┘
                    │
                    v
Step 2: Broker executes through Exchange
┌─────────────────────────────────────────────────┐
│ Broker.submit(order):                           │
│   - Exchange.execute(order)                     │
│   - Applies commission (0.1%)                   │
│   - Updates USD Wallet: $10,000 → $0            │
│   - Updates BTC Wallet: 0 → 0.0999 BTC          │
└─────────────────────────────────────────────────┘
                    │
                    v
Step 3: Observer creates next observation
┌─────────────────────────────────────────────────┐
│ Observer.observe():                             │
│   - Reads next values from DataFeed             │
│   - Shapes into observation array               │
│   - Returns: [close, rsi, trend, ...]           │
└─────────────────────────────────────────────────┘
                    │
                    v
Step 4: RewardScheme computes reward
┌─────────────────────────────────────────────────┐
│ RewardScheme.reward():                          │
│   - PBR: (price_change) × position              │
│   - Price went up $100, position is LONG        │
│   - Reward = +$100                              │
└─────────────────────────────────────────────────┘
                    │
                    v
Step 5: Stopper checks termination
┌─────────────────────────────────────────────────┐
│ Stopper.stop():                                 │
│   - Check: net_worth < 50% of initial?          │
│   - Check: reached end of data?                 │
│   - Returns: done = False (continue)            │
└─────────────────────────────────────────────────┘
                    │
                    v
Returns: (obs, reward, done, truncated, info)
```

---

## Core Components

### 1. Observer

**Purpose**: Converts market data into observations for the agent.

```
DataFeed ──────> Observer ──────> Agent
(streams)       (transforms)    (sees obs)
```

**Key file**: `tensortrade/env/default/observers.py`

```python
# TensorTradeObserver creates a window of observations
observer = TensorTradeObserver(
    feed=data_feed,
    window_size=10  # Agent sees last 10 time steps
)

# Observation shape: (window_size, num_features)
# Example: (10, 5) for 10 steps × 5 features
```

**What it does**:
- Reads from DataFeed streams
- Creates sliding window of history
- Defines observation_space for Gym

---

### 2. ActionScheme

**Purpose**: Converts agent's action (a number) into trading orders.

```
Agent ──────> ActionScheme ──────> Broker
(action=0)   (interprets)        (order)
```

**Key file**: `tensortrade/env/default/actions.py`

**Available schemes**:

| Scheme | Actions | Description |
|--------|---------|-------------|
| **BSH** | 0, 1 | Binary: Long (0) or Short/Cash (1) |
| **SimpleOrders** | 0 to N | Multiple sizes, sides, durations |
| **ManagedRiskOrders** | 0 to N | With stop-loss and take-profit |

**BSH in detail** (most common):

```python
class BSH(TensorTradeActionScheme):
    """
    Action space: Discrete(2)
    - Action 0: Be in BTC (long position)
    - Action 1: Be in USD (cash position)

    State machine:

    ┌─────────┐   action=1   ┌─────────┐
    │  LONG   │─────────────>│  CASH   │
    │  (BTC)  │<─────────────│  (USD)  │
    └─────────┘   action=0   └─────────┘

    Only trades when action differs from current position.
    """
```

**Why BSH causes overtrading**:
- Agent outputs action every step (every hour)
- Any flip between 0 and 1 triggers a trade
- Agent flips often → 2000+ trades/month

---

### 3. RewardScheme

**Purpose**: Computes the learning signal for the agent.

```
Portfolio ──────> RewardScheme ──────> Agent
(P&L)            (transforms)        (learns)
```

**Key file**: `tensortrade/env/default/rewards.py`

**Available schemes**:

| Scheme | Formula | Good For |
|--------|---------|----------|
| **PBR** | (price_change) × position | Direction learning |
| **SimpleProfit** | net_worth_change / net_worth | Simple benchmark |
| **RiskAdjustedReturns** | Sharpe or Sortino ratio | Risk-aware (doesn't work well) |
| **AdvancedPBR** | PBR + trade penalty + hold bonus | Experimental |

**PBR explained** (recommended):

```
PBR = Position-Based Returns

If price goes up $100:
  - Long position (+1): reward = +$100
  - Cash position (-1): reward = -$100

If price goes down $100:
  - Long position (+1): reward = -$100
  - Cash position (-1): reward = +$100

The agent learns: "be long when price will rise"
```

**Why PBR works better than SimpleProfit**:
- SimpleProfit rewards net worth increase
- But net worth only changes AFTER a trade
- PBR rewards being in the right position continuously
- Faster, more stable learning signal

---

### 4. Stopper

**Purpose**: Decides when to end an episode.

```python
class MaxLossStopper(Stopper):
    """Stop if net worth drops below threshold."""

    def stop(self, env):
        net_worth = env.portfolio.net_worth
        return net_worth < self.initial_worth * (1 - self.max_loss)
```

**Default behavior**:
- Stop if loss exceeds `max_allowed_loss` (e.g., 50%)
- Stop if data runs out

---

### 5. Portfolio and Exchange

**Purpose**: Simulate real trading execution.

```
Order ──────> Exchange ──────> Portfolio
             (commission)     (wallets)
```

```python
# Portfolio holds wallets
portfolio = Portfolio(USD, [
    Wallet(exchange, 10000 * USD),  # $10,000 cash
    Wallet(exchange, 0 * BTC),      # 0 BTC
])

# Exchange applies commission
exchange_options = ExchangeOptions(commission=0.001)  # 0.1%
exchange = Exchange("sim", execute_order, options=exchange_options)
```

---

### 6. DataFeed

**Purpose**: Reactive data pipeline for features.

```python
# Create streams from data
price = Stream.source(list(data["close"]), dtype="float").rename("price")
volume = Stream.source(list(data["volume"]), dtype="float").rename("volume")

# Derive features using stream operations
returns = price.pct_change().rename("returns")
rsi = price.apply(compute_rsi).rename("rsi")

# Combine into feed
feed = DataFeed([price, volume, returns, rsi])
feed.compile()
```

**Stream operations**:
- `.diff()` - difference from previous value
- `.pct_change()` - percentage change
- `.rolling(N).mean()` - moving average
- `.apply(fn)` - custom function

---

## Component Communication

```
┌────────────────────────────────────────────────────────────────────┐
│                         Data Flow                                  │
│                                                                    │
│   DataFeed ─────────┬───────────────────────> Observer             │
│   (streams)         │                         (observation)        │
│                     │                              │                │
│                     │                              v                │
│                     │                           Agent              │
│                     │                          (external)          │
│                     │                              │                │
│                     │                              v                │
│   Exchange <────────┼─── Broker <─────────── ActionScheme          │
│   (prices)          │   (orders)              (action→order)       │
│        │            │                                              │
│        v            │                                              │
│   Portfolio ────────┼───────────────────────> RewardScheme         │
│   (wallets)         │                         (computes reward)    │
│                     │                                              │
│   Clock ────────────┴───────────────────────> All Components       │
│   (timestep)                                  (synchronizes)       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Putting It Together

Here's how components are assembled in `train_simple.py`:

```python
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default

# 1. Data layer
price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
features = [Stream.source(list(data[c]), dtype="float").rename(c)
            for c in ['open', 'high', 'low', 'close', 'volume']]
feed = DataFeed(features)
feed.compile()

# 2. Execution layer
exchange_options = ExchangeOptions(commission=0.001)
exchange = Exchange("exchange", service=execute_order, options=exchange_options)(price)
cash = Wallet(exchange, 10000 * USD)
asset = Wallet(exchange, 0 * BTC)
portfolio = Portfolio(USD, [cash, asset])

# 3. Environment layer
reward_scheme = PBR(price=price)
action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

env = default.create(
    feed=feed,
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    window_size=10,
    max_allowed_loss=0.5
)
```

---

## Choosing Components

| If you want... | Use... |
|----------------|--------|
| Simple long/cash switching | BSH ActionScheme |
| Multiple position sizes | SimpleOrders ActionScheme |
| Stop-loss/take-profit | ManagedRiskOrders ActionScheme |
| Learn direction | PBR RewardScheme |
| Simple benchmark | SimpleProfit RewardScheme |
| Risk-adjusted | RiskAdjustedReturns (not recommended) |
| Reduce overtrading | AdvancedPBR (experimental) |

---

## Common Customizations

### Change commission

```python
exchange_options = ExchangeOptions(commission=0.005)  # 0.5%
```

### Change initial capital

```python
cash = Wallet(exchange, 50000 * USD)  # $50,000
```

### Change observation window

```python
env = default.create(
    ...
    window_size=20,  # Agent sees 20 time steps
)
```

### Change max allowed loss

```python
env = default.create(
    ...
    max_allowed_loss=0.3,  # Stop at 30% loss
)
```

---

## Key Takeaways

1. **TradingEnv** orchestrates all components in an episode loop
2. **Observer** creates what the agent sees
3. **ActionScheme** converts actions to orders (BSH is simplest)
4. **RewardScheme** creates the learning signal (PBR works best)
5. **Portfolio/Exchange** simulate real trading with commission
6. **DataFeed** provides reactive data streams

---

## Checkpoint

Before continuing, make sure you understand:

- [ ] What happens in each step of the episode loop?
- [ ] What does BSH ActionScheme do when action changes?
- [ ] Why is PBR better than SimpleProfit for learning?
- [ ] How does commission affect the Portfolio?

---

## Next Steps

[03-your-first-run.md](03-your-first-run.md) - Run train_simple.py and understand every output
