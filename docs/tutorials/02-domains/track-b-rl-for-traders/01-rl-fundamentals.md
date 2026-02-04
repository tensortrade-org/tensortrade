# RL Fundamentals for Traders

You understand markets, orders, and risk. This tutorial explains reinforcement learning in trading terms.

## Learning Objectives

After this tutorial, you will understand:
- How RL differs from traditional trading algorithms
- The key RL concepts in trading context
- Why RL for trading is challenging
- What makes a good reward function

---

## Traditional Algo Trading vs RL

### Traditional Approach

```python
# Rule-based strategy
def trade_signal(data):
    if rsi < 30 and macd_crossover:
        return "BUY"
    elif rsi > 70 or stop_loss_hit:
        return "SELL"
    return "HOLD"
```

**Characteristics**:
- Rules defined by humans
- Based on intuition and backtesting
- Fixed parameters (RSI=30, RSI=70)
- Doesn't adapt to changing markets

### RL Approach

```python
# RL-based strategy
def trade_signal(data):
    state = extract_features(data)  # RSI, MACD, etc.
    action = agent.predict(state)   # Neural network decision
    return action  # 0=BUY, 1=SELL
```

**Characteristics**:
- Rules discovered from data
- Learns from trial and error
- Adapts parameters through training
- Can find non-obvious patterns

---

## The RL Framework

Think of RL as training a junior trader:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "Junior Trader" (Agent)                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │   Brain: Neural Network                                 │  │
│   │   ┌─────────────────────────────────────────────────┐  │  │
│   │   │  Input: Market data (RSI=45, trend=up, etc.)   │  │  │
│   │   │  Output: Decision (BUY, SELL, HOLD)            │  │  │
│   │   │  Learning: Adjust weights based on P&L         │  │  │
│   │   └─────────────────────────────────────────────────┘  │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   "Market" (Environment)                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │   Shows: Current market state (prices, indicators)     │  │
│   │   Accepts: Trading decisions from agent                │  │
│   │   Returns: Profit/loss from each decision              │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   "Performance Review" (Reward)                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                         │  │
│   │   After each decision:                                  │  │
│   │   - Good decision (profit): positive feedback          │  │
│   │   - Bad decision (loss): negative feedback             │  │
│   │   Agent adjusts behavior to get more positive feedback │  │
│   │                                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key RL Terms in Trading Context

| RL Term | Trading Equivalent |
|---------|-------------------|
| **State** | Current market snapshot (prices, indicators, position) |
| **Action** | Trading decision (buy, sell, hold) |
| **Reward** | Profit/loss signal after action |
| **Policy** | The trading strategy the agent learns |
| **Episode** | One complete trading period (e.g., 1 month) |
| **Environment** | Simulated market with your data |
| **Agent** | The neural network making decisions |

### State (What the Agent Sees)

```python
state = [
    0.45,   # RSI (normalized)
    0.23,   # Returns last 1h
    -0.12,  # Returns last 24h
    0.67,   # Trend strength
    1.0,    # Current position (1=long, -1=cash)
]
```

The agent only knows what's in the state. It can't "see" the future.

### Action (What the Agent Decides)

In TensorTrade's BSH scheme:
- Action 0: Go long (hold BTC)
- Action 1: Go cash (hold USD)

The agent outputs a number, the ActionScheme converts it to actual orders.

### Reward (Learning Signal)

```
This is CRITICAL. The reward shapes what the agent learns.

Bad reward: Final P&L only
  - Agent gets one signal at end of episode
  - Very slow learning

Good reward: Step-by-step P&L (PBR)
  - Agent gets signal every step
  - Fast, stable learning
```

---

## The Learning Process

### 1. Exploration Phase

Early in training, the agent tries random actions:

```
Step 1: [RSI=25, trend=up] → BUY  → +$50   (lucky!)
Step 2: [RSI=30, trend=up] → SELL → -$20   (oops)
Step 3: [RSI=28, trend=up] → BUY  → +$30   (learning: low RSI + up trend = buy)
```

### 2. Exploitation Phase

Later, the agent uses what it learned:

```
Step 100: [RSI=25, trend=up] → BUY  (learned this pattern is good)
Step 101: [RSI=75, trend=down] → SELL (learned this pattern is bad for longs)
```

### 3. The Policy

The policy is the learned mapping: state → action

```
Neural Network (Policy)
Input: [RSI, trend, momentum, ...] (5-20 features)
Hidden: [64 neurons] → [64 neurons]
Output: [probability of BUY, probability of SELL]
```

---

## PPO: The Algorithm We Use

PPO (Proximal Policy Optimization) is the RL algorithm. Think of it as the "training method."

**Key PPO concepts for traders**:

| Parameter | Meaning | Trading Impact |
|-----------|---------|----------------|
| **learning_rate** | How fast to update weights | Too high = unstable, too low = slow |
| **gamma** | How much to value future rewards | High = patient trader, Low = short-term |
| **entropy** | How much to explore vs exploit | High = more experimentation |
| **clip_param** | Limits how much policy can change | Prevents wild strategy shifts |

**Best values from experiments**:
```python
lr = 3.29e-05      # Very slow learning (stable)
gamma = 0.992      # Values rewards ~100 steps out
entropy = 0.015    # Mostly exploit, little explore
clip = 0.123       # Moderate policy changes allowed
```

---

## Why RL for Trading is Hard

### 1. Non-Stationary Environment

Markets change. What worked in 2020 may not work in 2024.

```
Training data: Bull market (2020-2021)
Test data: Bear market (2022)
Result: Agent learned "always buy" → disaster
```

**Solution**: Use scale-invariant features, train on diverse conditions.

### 2. Delayed Rewards

A trade might be "right" but take weeks to pay off.

```
Agent buys at $100,000
Price drops to $95,000 (agent sees negative reward)
Agent sells (learned "buying was bad")
Price rises to $110,000 (agent missed the real profit)
```

**Solution**: High gamma (0.99+) to value future rewards.

### 3. Reward Hacking

Agents find unexpected ways to maximize reward.

```
Reward: Maximize total profit
Agent learns: Trade constantly (each profitable micro-trade adds up)
Result: 2000 trades/month, commission destroys real profit
```

**Solution**: Include commission in simulation, penalize trading.

### 4. Overfitting

Agent memorizes training data patterns that don't generalize.

```
Training: Agent learns "Wednesday 3pm = always buy"
         (Happened to work in training data)
Test: Pattern doesn't exist in new data
Result: Random-looking trading, losses
```

**Solution**: Fewer features, smaller networks, early stopping.

---

## What Makes a Good Reward Function

### Bad: Final P&L

```python
def reward(env):
    if episode_done:
        return portfolio.net_worth - 10000
    return 0
```

**Problem**: Agent gets ONE signal per episode. Very slow learning.

### Better: Simple Profit

```python
def reward(env):
    return portfolio.net_worth_change / portfolio.net_worth
```

**Problem**: Only non-zero when trades happen. Agent may not learn to hold.

### Best: Position-Based Returns (PBR)

```python
def reward(env):
    return price_change * position  # +1 for long, -1 for cash
```

**Why it works**:
- Signal every step, not just on trades
- Rewards being in the RIGHT position
- Symmetric: rewards avoiding losses too
- Fast, stable learning

---

## The Credit Assignment Problem

When something goes wrong, what caused it?

```
Step 1: BUY at $100,000
Step 2: HOLD
Step 3: HOLD
Step 4: HOLD
Step 5: Market crashes, -$5,000

Which step was the mistake?
- Buying at step 1? (couldn't predict crash)
- Holding at step 4? (should have sold earlier)
- None? (unpredictable event)
```

RL handles this through:
1. **Temporal difference learning** - Propagate rewards backward
2. **Value function** - Estimate future rewards from each state
3. **Advantage estimation** - Compare actual vs expected outcomes

You don't need to implement this - PPO handles it.

---

## Practical Checklist

Before training, verify:

- [ ] **State includes enough info**: Agent can't learn from nothing
- [ ] **Features are normalized**: Neural networks like [-1, 1] range
- [ ] **Reward is dense**: Signal every step, not just at end
- [ ] **Commission is realistic**: Otherwise agent learns unrealistic strategy
- [ ] **Training data is long enough**: At least 1000+ steps per episode
- [ ] **Validation set exists**: To detect overfitting

---

## Key Takeaways

1. **RL learns strategies from data** rather than following fixed rules
2. **The reward function is critical** - PBR works much better than final P&L
3. **PPO parameters matter** - Low learning rate, high gamma for trading
4. **Overfitting is the main enemy** - Use simpler models, validate properly
5. **Markets are non-stationary** - What worked before may not work again

---

## Checkpoint

Before continuing, make sure you understand:

- [ ] Why PBR gives better learning signals than final P&L
- [ ] What gamma controls and why it should be high for trading
- [ ] Why overfitting is the main challenge
- [ ] What "policy" means in RL context

---

## Next Steps

[02-common-failures.md](02-common-failures.md) - The critical pitfalls that destroy RL trading agents
