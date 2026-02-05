# Overfitting: Detection and Prevention

Overfitting is the default failure mode of RL trading agents. This tutorial explains how to detect it and prevent it.

## Learning Objectives

After this tutorial, you will understand:
- What overfitting looks like in trading RL
- How to detect it early
- Proven prevention techniques
- Why simpler models often work better

---

## What is Overfitting?

Overfitting occurs when your model memorizes the training data instead of learning generalizable patterns.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   UNDERFITTING          GOOD FIT            OVERFITTING        │
│                                                                 │
│   Model too simple     Model captures       Model memorizes    │
│                        real patterns        noise              │
│                                                                 │
│   ~~~~                  ─╲__╱─              ╱╲╱╲╱╲╱╲          │
│   ────                    ╱╲                ╱      ╲          │
│                                                                 │
│   Train: BAD            Train: GOOD         Train: EXCELLENT   │
│   Test: BAD             Test: GOOD          Test: TERRIBLE     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Overfitting in Trading RL

### What the Agent Learns (Overfitting)

```python
# Agent memorizes specific patterns:
"On training day 47 at hour 14, price went up after a specific
 combination of RSI=32.4 and volume=1,847 BTC"

# Agent thinks this is a rule:
"When RSI ≈ 32.4 and volume ≈ 1,847, BUY"

# But in reality, it was random
# Pattern doesn't exist in test data
```

### What We Want (Generalization)

```python
# Agent learns general patterns:
"When RSI is low AND price is below moving average AND
 volume is increasing, buying tends to work"

# This pattern generalizes to new data
```

---

## Detecting Overfitting

### Signal 1: Training/Validation Gap

```
Iteration   Train P&L   Val P&L    Gap        Status
────────────────────────────────────────────────────
    10      $+500       $+200      $300       Normal
    20      $+1,500     $+400      $1,100     Warning
    30      $+3,000     $+100      $2,900     OVERFITTING
    40      $+5,000     $-200      $5,200     SEVERE
    50      $+7,000     $-600      $7,600     STOP!
```

**Rule of thumb**: If train P&L is 5x+ better than validation, you're overfitting.

### Signal 2: Validation Starts Declining

```
                Train P&L           Val P&L
                    │                   │
                    │    ────────────>  │  ╭─────╮
                    │   ╱               │ ╱       ╲
                    │  ╱                │╱         ╲
                    │ ╱                 │           ╲
                    │╱                  │            ╲
────────────────────┼───────────────────┼─────────────────▶ Iteration
                    │                   │
                    │   OVERFITTING     │
                    │   STARTS HERE ────┼─────────
                    │                   │
```

### Signal 3: Reward Increases But P&L Doesn't

```python
# During training callback:
reward = episode.total_reward     # Learning signal
pnl = portfolio.net_worth - 10000  # Actual money

# If reward ↑ but pnl flat or ↓, agent is gaming reward
```

---

## Prevention Techniques

### 1. Smaller Networks

```python
# OVERFITS (can memorize patterns)
model={"fcnet_hiddens": [256, 256]}
# 256 × 256 = 65,536 parameters in first layer alone

# GENERALIZES BETTER
model={"fcnet_hiddens": [64, 64]}
# 64 × 64 = 4,096 parameters

# From our experiments:
# 256x256: Test P&L -$2,690
# 128x128: Test P&L -$650
```

### 2. Fewer Features

```python
# OVERFITS (too many features to memorize)
features = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'atr_7', 'atr_14', 'atr_21',
    ...  # 34 features
]

# GENERALIZES BETTER (only essential features)
features = [
    'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h',  # Returns
    'rsi',                                       # Momentum
    'trend_20', 'trend_50',                     # Trend
    'vol_norm',                                 # Volatility
]  # 8 features

# From our experiments:
# 34 features: Test P&L -$2,690
# 13 features: Test P&L -$650
```

### 3. Early Stopping

```python
best_val_pnl = float('-inf')
patience = 5
no_improvement = 0

for i in range(100):
    algo.train()

    if (i + 1) % 10 == 0:
        val_pnl = evaluate(algo, val_data)

        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            algo.save('/tmp/best_model')
            no_improvement = 0
        else:
            no_improvement += 1

        # Early stop if no improvement for 5 evals
        if no_improvement >= patience:
            print(f"Early stopping at iteration {i+1}")
            break

# Use best model, not final model
algo.restore('/tmp/best_model')
```

### 4. Higher Entropy

```python
# Low entropy: Agent commits to one strategy (can overfit)
entropy_coeff=0.001

# Higher entropy: Agent keeps exploring (prevents overfitting)
entropy_coeff=0.05

# Trade-off: Higher entropy = slower convergence but better generalization
```

### 5. Regularization

```python
# Add L2 regularization to network weights
# (Penalizes large weights that memorize patterns)

# In Ray RLlib, use:
.training(
    model={
        "fcnet_hiddens": [64, 64],
        # Custom model with regularization
    }
)
```

### 6. More Training Data

```python
# OVERFITS (too little data)
train_data = data.tail(500)  # ~20 days

# GENERALIZES BETTER (more diverse patterns)
train_data = data.tail(4000)  # ~167 days

# More data = more patterns = harder to memorize
```

### 7. Data Augmentation

```python
# Add noise to training data
def augment(df):
    df = df.copy()
    noise = np.random.normal(0, 0.001, len(df))
    df['close'] = df['close'] * (1 + noise)
    return add_features(df)

# Train on multiple augmented versions
for _ in range(5):
    train_on(augment(train_data))
```

---

## The Overfitting-Underfitting Trade-off

```
                    │
    Test            │
    Performance     │              ╭─────╮
                    │            ╱         ╲
                    │          ╱             ╲
                    │        ╱                 ╲
                    │      ╱                     ╲
                    │    ╱                         ╲
                    │  ╱                             ╲
                    │╱                                 ╲
────────────────────┼───────────────────────────────────────▶
                    │          Model Complexity
                    │
                    │  ◀── Underfitting │ Overfitting ──▶
                    │
                    │  SWEET SPOT = somewhere in middle
```

### Finding the Sweet Spot

1. **Start simple** - Small network, few features
2. **Gradually add complexity** - Larger network, more features
3. **Monitor validation** - Stop when validation stops improving
4. **Use the model that generalizes best** - Not the one with best training

---

## Practical Checklist

### Before Training

- [ ] Features are scale-invariant (not raw prices)
- [ ] Network is reasonably small (start with [64, 64])
- [ ] Feature count is limited (start with 5-10)
- [ ] Training data is long enough (1000+ candles)
- [ ] Validation set is properly separated (no overlap)

### During Training

- [ ] Log both training AND validation metrics
- [ ] Compare train/val gap every 10 iterations
- [ ] Stop training when validation stops improving
- [ ] Save best validation model, not final model

### After Training

- [ ] Evaluate on held-out TEST data (not validation)
- [ ] Compare to baseline (Buy-and-Hold)
- [ ] Check if results make sense (not too good to be true)

---

## Case Study: Our Experiments

### Experiment 1: Severe Overfitting

```
Configuration:
  - 34 features
  - Network: [256, 256]
  - Training: 150 iterations

Results:
  - Training P&L: +$6,366 (AMAZING!)
  - Test P&L: -$2,690 (DISASTER)
  - Agent memorized patterns that don't exist
```

### Experiment 5: Better Generalization

```
Configuration:
  - 13 features (scale-invariant)
  - Network: [128, 128]
  - Training: 100 iterations with early stopping

Results:
  - Training P&L: +$800
  - Validation P&L: -$125 (beat B&H!)
  - Test P&L: -$650
  - Still not profitable, but MUCH better generalization
```

### Lesson

```
Better to have:
  Training: $+800, Test: $-650    (small gap)

Than:
  Training: $+6,366, Test: $-2,690  (huge gap)

The smaller gap means the model LEARNED something real.
```

---

## Why Trading RL is Especially Prone to Overfitting

1. **Markets are non-stationary** - Patterns change over time
2. **Noise looks like signal** - Random movements can appear meaningful
3. **Sample sizes are small** - Even years of data is limited
4. **Rewards are delayed** - Hard to learn which actions mattered
5. **Distribution shift** - Test data is from different market conditions

### The Hard Truth

Even with perfect overfitting prevention, trading RL is hard:
- Our best model: +$239 at 0% commission
- But still loses at realistic commission
- Overfitting is only ONE of the challenges

---

## Key Takeaways

1. **Overfitting is the default** - Assume it's happening until proven otherwise
2. **Monitor validation P&L** - Not just training metrics
3. **Smaller is often better** - Fewer features, smaller networks
4. **Early stop** - Use best validation model, not final model
5. **Train/test gap tells the story** - Large gap = overfitting

---

## Checkpoint

After reading this tutorial, verify you can:

- [ ] Identify overfitting from train/val metrics
- [ ] List 3 techniques to prevent overfitting
- [ ] Explain why smaller networks can be better
- [ ] Set up early stopping in training loop

---

## Next Steps

[02-commission.md](02-commission.md) - The commission problem (our key discovery)
