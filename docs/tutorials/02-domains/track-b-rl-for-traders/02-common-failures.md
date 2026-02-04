# Common RL Trading Failures

This is the most important tutorial. These failures have destroyed countless trading agents. Learn them before you train.

## The Bottom Line

After 100+ Optuna trials and 8 major experiments:

| Metric | Result |
|--------|--------|
| Agent (0% commission) | **+$239 profit** (beats B&H by $594) |
| Agent (0.1% commission) | **-$650 loss** (loses to B&H by $295) |
| Buy-and-Hold | -$355 |

**The agent CAN predict direction. Commission costs destroy the profit.**

---

## Failure #1: Overfitting (The Silent Killer)

### What Happens

```
Training Reward vs Validation Performance

Iteration 10:   Train +$500,   Val -$200    (learning)
Iteration 50:   Train +$3,000, Val +$100    (generalizing)
Iteration 100:  Train +$6,000, Val -$800    (OVERFITTING!)
Iteration 150:  Train +$8,000, Val -$2,000  (memorizing noise)

                Training ████████████████████████████████
Iter 50:        Validation ████████
                Training ████████████████████████████████████████████████
Iter 150:       Validation ██   ← DIVERGING = OVERFITTING
```

### Why It Happens

The agent memorizes specific patterns in training data:
- "Tuesday at 2pm, price always goes up" (coincidence in training)
- "When RSI hits exactly 32.7, buy" (noise, not signal)
- "After pattern X-Y-Z, sell" (happened 3 times, agent memorizes)

### How to Detect

```python
# Track validation performance during training
for i in range(iterations):
    algo.train()

    if i % 10 == 0:
        train_pnl = evaluate(algo, train_data)
        val_pnl = evaluate(algo, val_data)

        # WARNING: Overfitting signal
        if train_pnl > prev_train_pnl and val_pnl < prev_val_pnl:
            print("WARNING: Training improving but validation declining!")
```

### How to Prevent

| Technique | Implementation |
|-----------|----------------|
| **Smaller network** | `[64, 64]` not `[256, 256]` |
| **Fewer features** | 5-13 features, not 34 |
| **Early stopping** | Stop when validation stops improving |
| **Regularization** | High entropy coefficient (0.01-0.05) |
| **More training data** | 4000+ candles minimum |

**From experiments**:
```python
# OVERFITS (Experiment 1)
model = {"fcnet_hiddens": [256, 256]}
features = 34
result: Train +$6,366, Test -$2,690

# GENERALIZES BETTER (Experiment 5)
model = {"fcnet_hiddens": [128, 128]}
features = 13
result: Train +$800, Test -$650  # Much smaller gap
```

---

## Failure #2: Overtrading (The Profit Destroyer)

### What Happens

```
Agent's Trading Behavior

Hour 1:  BUY   │  Commission: -$10
Hour 2:  SELL  │  Commission: -$10
Hour 3:  BUY   │  Commission: -$10
Hour 4:  HOLD  │
Hour 5:  SELL  │  Commission: -$10
...
Hour 720: SELL │  Commission: -$10

Total trades: 2,000
Direction prediction profit: +$239
Commission paid: -$2,000
Net result: -$1,761 LOSS
```

### The Math

```
Trades per month:          ~2,000
Commission per trade:      0.1%
Account size:              $10,000
Average trade size:        $10,000 (full position)

Commission cost = 2000 × 0.1% × $10,000 ÷ 100 = $2,000/month

To break even, agent needs: +20% monthly returns
Just to cover commission costs!
```

### Why It Happens

1. **BSH requires trades to change position**
   - Agent wants to be long → outputs 0
   - Agent wants to be cash → outputs 1
   - Every change = one trade

2. **Agent is uncertain**
   - Slightly prefers long → 0
   - Next step, slightly prefers cash → 1
   - Oscillates back and forth

3. **No penalty for trading**
   - PBR rewards position, not trading frequency
   - Agent sees no downside to frequent changes

### How to Detect

```python
# Count trades during evaluation
trades = 0
prev_action = None
for step in range(episode_length):
    action = agent.predict(obs)
    if action != prev_action:
        trades += 1
    prev_action = action

trades_per_day = trades / (episode_length / 24)
print(f"Trades per day: {trades_per_day}")

# HEALTHY: 1-5 trades/day
# WARNING: 10-20 trades/day
# CRITICAL: 50+ trades/day
```

### How to Reduce

| Technique | How |
|-----------|-----|
| **Train with commission** | Agent feels the cost |
| **Higher training commission** | 0.3-0.5% during training |
| **Trade penalty in reward** | Subtract for position changes |
| **Action masking** | Prevent trading for N steps after trade |
| **Position sizing** | Replace BSH with continuous actions |

**From experiments**:
```python
# Training commission affects trading frequency
train_commission = 0.0%   → Agent trades constantly
train_commission = 0.1%   → Agent trades frequently
train_commission = 0.3%   → Agent trades moderately
train_commission = 0.5%   → Agent trades less, better direction

# Results (Test P&L with 0% commission):
Train @ 0.0%: -$102
Train @ 0.3%: +$167
Train @ 0.5%: +$239  # Best direction prediction
```

---

## Failure #3: Wrong Reward Function

### Failure: Final P&L Only

```python
# BAD
def reward(env):
    if episode_done:
        return portfolio.net_worth - initial_cash
    return 0

# Agent gets ONE number after 720 steps
# Learning is incredibly slow
# Agent can't tell which actions were good
```

**Symptom**: Reward doesn't improve even after many iterations.

### Failure: Risk-Adjusted Returns

```python
# BAD for RL
def reward(env):
    returns = portfolio.returns()
    sharpe = returns.mean() / returns.std()
    return sharpe
```

**Why it fails**:
- Sharpe ratio needs many samples to be meaningful
- Single-step Sharpe is noisy nonsense
- Agent learns nothing useful

**From experiments**:
```
Sharpe ratio reward: Test P&L -$3,174 (MUCH WORSE than PBR)
PBR reward:          Test P&L -$650
```

### Success: PBR (Position-Based Returns)

```python
# GOOD
def reward(env):
    price_change = current_price - previous_price
    position = +1 if holding_btc else -1
    return price_change * position

# Agent gets immediate feedback every step
# Clear signal: "you were in the right/wrong position"
```

---

## Failure #4: Data Leakage

### What It Looks Like

```python
# LEAK: Using future information
df['tomorrow_return'] = df['close'].pct_change().shift(-1)  # FUTURE DATA!
df['target'] = (df['tomorrow_return'] > 0).astype(int)       # LABEL LEAK!

# Agent learns to "predict" using future it shouldn't see
# Training looks amazing, real trading fails completely
```

### Common Leaks

| Leak Type | Example |
|-----------|---------|
| **Look-ahead bias** | Using tomorrow's data for today's features |
| **Label leakage** | Target variable in features |
| **Wrong split** | Random split (should be time-based) |
| **Overlap** | Validation overlaps with training |

### How to Prevent

```python
# CORRECT time-based split
train_data = data.iloc[:split_idx]        # Past only
val_data = data.iloc[split_idx:split_idx2] # Future from train
test_data = data.iloc[split_idx2:]         # Future from val

# NEVER shuffle time series data
# NEVER use .shift(-1) for features
# ALWAYS simulate real-time: agent only sees past
```

---

## Failure #5: Non-Stationary Features

### What Happens

```
Training data (2020):
  - BTC price: $10,000 - $30,000
  - Volume: 500-2000 BTC/hour

Test data (2024):
  - BTC price: $60,000 - $100,000  # 3x higher!
  - Volume: 2000-8000 BTC/hour     # 4x higher!

Agent learned: "price > $50,000 means expensive, sell"
Reality: $50,000 is now cheap
```

### Solution: Scale-Invariant Features

```python
# BAD: Raw values
features = [price, volume, sma_20]

# GOOD: Normalized/relative values
features = [
    returns_1h,              # Percentage, not absolute
    (price - sma_20) / sma_20,  # Relative to average
    volume / volume_sma_20,  # Ratio, not absolute
    np.tanh(returns * 10),   # Bounded to [-1, 1]
]
```

**From experiments**:
```python
# Best features (scale-invariant)
'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'ret_48h',  # Returns
'rsi',                                                  # Already normalized
'trend_20', 'trend_50', 'trend_strength',              # Relative to SMA
'vol_norm', 'vol_ratio',                               # Normalized
'bb_pos'                                               # Bounded 0-1
```

---

## Failure #6: Reward Hacking

### Example: Gaming Sharpe Ratio

```
Agent discovers: "If I only trade once and make $1 profit,
                  my Sharpe ratio is infinite!"

Reward: Very high (Sharpe = profit / 0 risk)
Reality: Made $1 total
```

### Example: Gaming Trade Count

```
Reward: profit - 0.001 * trade_count

Agent discovers: "If I never trade, I never get penalized!"

Behavior: Agent holds initial position forever
Reality: Misses all opportunities
```

### Prevention

1. **Test reward function logic before training**
2. **Watch for unexpected behavior patterns**
3. **Validate that high reward = high P&L**
4. **Use multiple metrics (P&L, Sharpe, Max Drawdown)**

---

## Failure #7: Insufficient Training Data

### The Problem

```
Training data: 200 candles (8 days)
Patterns the agent can learn: Very limited
Overfitting risk: Very high

Training data: 4000 candles (167 days)
Patterns the agent can learn: Many market conditions
Overfitting risk: Much lower
```

### Minimum Data Guidelines

| Timeframe | Minimum Candles | ~Days |
|-----------|-----------------|-------|
| 1 hour | 4000+ | 167 |
| 4 hour | 1000+ | 167 |
| 1 day | 500+ | 500 |

### Why More Data Helps

```
With 200 candles:
  - Maybe 1 significant move up
  - Maybe 1 significant move down
  - Agent sees limited scenarios

With 4000 candles:
  - Multiple bull runs
  - Multiple corrections
  - Sideways periods
  - Flash crashes
  - Agent sees diverse scenarios
```

---

## Quick Diagnostic Checklist

When your agent fails, check these in order:

### 1. Is it overfitting?
```
Training P&L >> Validation P&L?
→ Yes: Reduce model size, features, training iterations
```

### 2. Is it overtrading?
```
Trades per day > 10?
→ Yes: Increase training commission, add trade penalty
```

### 3. Is the reward function working?
```
Reward improving but P&L not?
→ Check reward logic, try PBR
```

### 4. Are features leaking future?
```
Training accuracy > 80%?
→ Suspicious, check for look-ahead bias
```

### 5. Are features non-stationary?
```
Train/test price ranges very different?
→ Use scale-invariant features only
```

### 6. Is there enough data?
```
Training candles < 1000?
→ Get more data or use larger timeframe
```

---

## The Path to Profitability

Based on all experiments, here's what actually works:

### Current Best Configuration

```python
# Hyperparameters (from Optuna)
lr = 3.29e-05
gamma = 0.992
entropy = 0.015
clip = 0.123
hidden = [128, 128]
window = 17

# Training
train_commission = 0.3% - 0.5%  # Teach trading discipline
features = 13  # Scale-invariant only
iterations = 100  # With early stopping on validation

# Result: +$239 direction prediction (0% commission)
```

### What's Still Needed

The agent predicts direction correctly but trades too much:

| Solution | Status |
|----------|--------|
| Position sizing (not binary) | Not implemented |
| Confidence threshold | Not implemented |
| Minimum hold period | Not implemented |
| Trade-aware reward | Experimental (AdvancedPBR) |

---

## Key Takeaways

1. **Overfitting is the default** - Assume it's happening until proven otherwise
2. **Commission destroys naive strategies** - Train with realistic costs
3. **PBR >> other rewards** - For trading, position-based returns work best
4. **Scale-invariant features only** - Raw prices don't generalize
5. **More data + smaller models** - Better than less data + bigger models
6. **The agent CAN predict direction** - The problem is trading frequency

---

## Checkpoint

Before training your own agent, verify you understand:

- [ ] How to detect overfitting (training >> validation)
- [ ] Why overtrading destroys profits (commission math)
- [ ] Why PBR works better than Sharpe/final P&L
- [ ] What scale-invariant features are
- [ ] The current best configuration and its results

---

## Next Steps

You've learned the pitfalls. Now learn the components:
- [Action Schemes](../../03-components/01-action-schemes.md)
- [Reward Schemes](../../03-components/02-reward-schemes.md)
- [First Training](../../04-training/01-first-training.md)
