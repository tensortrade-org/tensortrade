# The Three Pillars of TensorTrade

TensorTrade sits at the intersection of three complex domains. Understanding how they connect is essential before diving into code.

## Learning Objectives

After reading this tutorial, you will understand:
- What reinforcement learning brings to trading
- What trading concepts you need for RL
- How TensorTrade bridges these domains

---

## The Three Domains

```
        ┌─────────────────────┐
        │   Reinforcement     │
        │      Learning       │
        │                     │
        │  • Agent learns     │
        │  • From rewards     │
        │  • Through actions  │
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    v              v              v
┌───────────┐ ┌───────────┐ ┌───────────┐
│  Trading  │ │TensorTrade│ │   Data    │
│  Domain   │ │           │ │Engineering│
│           │ │  Bridges  │ │           │
│ • Markets │ │   all 3   │ │ • OHLCV   │
│ • Orders  │ │  domains  │ │ • Features│
│ • Risk    │ │           │ │ • Streams │
└───────────┘ └───────────┘ └───────────┘
```

---

## Pillar 1: Reinforcement Learning

RL is about learning through trial and error. An **agent** takes **actions** in an **environment** and receives **rewards**.

### The RL Loop

```
┌─────────────────────────────────────────┐
│                                         │
│   Environment                           │
│   ┌─────────────────────────────────┐  │
│   │                                 │  │
│   │  State: [price, rsi, trend...]  │  │
│   │                                 │  │
│   └──────────────┬──────────────────┘  │
│                  │                      │
│                  v                      │
│   ┌─────────────────────────────────┐  │
│   │  Agent sees state, picks action │  │
│   │  Action: BUY / SELL / HOLD      │  │
│   └──────────────┬──────────────────┘  │
│                  │                      │
│                  v                      │
│   ┌─────────────────────────────────┐  │
│   │  Environment returns reward     │  │
│   │  Reward: +$50 or -$30           │  │
│   └──────────────┬──────────────────┘  │
│                  │                      │
│                  v                      │
│   Agent updates its policy to get      │
│   more rewards in the future           │
│                                         │
└─────────────────────────────────────────┘
```

### Key RL Concepts for Trading

| Concept | In Trading |
|---------|------------|
| **State** | Market data: price, volume, indicators |
| **Action** | Trade decision: buy, sell, hold |
| **Reward** | Profit/loss from the action |
| **Policy** | The strategy the agent learns |
| **Episode** | One complete trading period |

### Why RL for Trading?

Traditional algorithmic trading uses fixed rules:
```
IF RSI < 30 THEN BUY
IF RSI > 70 THEN SELL
```

RL discovers rules from data:
```
Agent learns: "When RSI < 30 AND volume is rising AND
trend is up, buying has historically led to profit"
```

The agent can discover patterns humans might miss.

---

## Pillar 2: Trading Domain

You don't need to be a quant, but you need these fundamentals.

### The Order Management System (OMS)

TensorTrade simulates a real trading system:

```
┌─────────────────────────────────────────────────┐
│                    Portfolio                     │
│  ┌──────────────┐      ┌──────────────────────┐│
│  │ USD Wallet   │      │ BTC Wallet           ││
│  │ $10,000      │      │ 0.0 BTC              ││
│  └──────────────┘      └──────────────────────┘│
│                                                 │
│  Net Worth = USD + (BTC × Current Price)       │
└─────────────────────────────────────────────────┘
          │
          │ Agent says: BUY
          v
┌─────────────────────────────────────────────────┐
│                    Exchange                      │
│                                                 │
│  Order: BUY 0.1 BTC @ $100,000                  │
│  Commission: 0.1% = $10                         │
│                                                 │
│  Execution:                                     │
│    USD Wallet: $10,000 - $10,010 = -$10        │
│    BTC Wallet: 0.0 + 0.0999 = 0.0999 BTC       │
└─────────────────────────────────────────────────┘
```

### Key Trading Concepts

| Concept | Definition | In TensorTrade |
|---------|------------|----------------|
| **Position** | What you own | Cash, BTC, or both |
| **Commission** | Trading fee | 0.1% typical |
| **Net Worth** | Total value | USD + BTC value |
| **P&L** | Profit & Loss | Final - Initial |
| **Slippage** | Price movement during execution | Simulated |

### The Commission Problem

This is crucial. From our experiments:

```
Agent trades 2,000 times per month
Each trade costs 0.1% commission
2,000 × 0.1% × $10,000 = $2,000 in fees

Agent's direction prediction profit: +$239
Commission cost: -$2,000
Net result: -$1,761 LOSS
```

**The agent can predict direction, but trades too much.**

---

## Pillar 3: Data Engineering

RL agents learn from features. Better features = better learning.

### OHLCV Data

The foundation of all trading data:

```
┌─────────────────────────────────────────────────┐
│                 One Candle (1 hour)             │
│                                                 │
│  Open:   $100,000  (price at start)            │
│  High:   $101,500  (highest price)             │
│  Low:    $99,200   (lowest price)              │
│  Close:  $100,800  (price at end)              │
│  Volume: 1,234 BTC (amount traded)             │
└─────────────────────────────────────────────────┘
```

### Feature Engineering

Raw OHLCV data isn't enough. The agent needs derived features:

```python
# Returns: How much did price change?
returns_1h = (close - prev_close) / prev_close

# RSI: Is market overbought/oversold?
rsi = compute_rsi(close, period=14)

# Trend: Is price above/below average?
trend = (close - sma_50) / sma_50
```

### Scale-Invariant Features

**Critical insight from experiments**: Raw prices don't generalize.

```
BAD:  price = $100,000  (depends on when you look)
GOOD: returns = +2.5%   (works at any price level)

BAD:  volume = 1,234 BTC  (varies wildly)
GOOD: volume_ratio = current / 20-day average  (normalized)
```

Our best models use 13 scale-invariant features, not 34 raw indicators.

---

## How TensorTrade Connects Everything

```
┌────────────────────────────────────────────────────────────────┐
│                       TensorTrade                              │
│                                                                │
│  DATA LAYER                                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  DataFeed: OHLCV + Derived Features                      │ │
│  │  Stream API for reactive data processing                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              v                                 │
│  ENVIRONMENT LAYER                                             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  TradingEnv: Gym-compatible environment                  │ │
│  │  • Observer: Creates state from DataFeed                 │ │
│  │  • ActionScheme: Converts actions to orders              │ │
│  │  • RewardScheme: Computes learning signal                │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              v                                 │
│  EXECUTION LAYER                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  OMS: Simulates real trading                             │ │
│  │  • Exchange: Price feed + commission                     │ │
│  │  • Broker: Order execution                               │ │
│  │  • Portfolio: Wallet management                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Where Are You Coming From?

Your background determines what you need to learn:

### Coming from RL?

You know agents, rewards, and policies. You need:
- How orders and commissions work
- Why slippage matters
- The difference between P&L and reward
- See: [Track A: Trading for RL People](../02-domains/track-a-trading-for-rl/01-trading-basics.md)

### Coming from Trading/Quant?

You know markets and orders. You need:
- How RL agents learn from rewards
- Why reward shaping is tricky
- Common RL failure modes
- See: [Track B: RL for Traders](../02-domains/track-b-rl-for-traders/01-rl-fundamentals.md)

### New to Both?

Start with the basics of each:
- See: [Track C: Full Introduction](../02-domains/track-c-full-intro/README.md)

---

## Key Takeaways

1. **RL learns strategies from data** instead of following fixed rules
2. **Trading has real costs** (commission) that destroy naive strategies
3. **Feature engineering is critical** - use scale-invariant features
4. **TensorTrade bridges all three** with a modular component system

---

## Checkpoint

Before continuing, make sure you understand:

- [ ] What is the RL loop (state → action → reward)?
- [ ] What is commission and why does it matter?
- [ ] Why are scale-invariant features better than raw prices?
- [ ] What are the main TensorTrade layers (Data, Environment, Execution)?

---

## Next Steps

[02-architecture.md](02-architecture.md) - Deep dive into TensorTrade's component system
