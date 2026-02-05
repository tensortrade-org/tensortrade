# Walk-Forward Validation

Proper validation is critical for trading strategies. This tutorial covers walk-forward methodology.

## Learning Objectives

After this tutorial, you will understand:
- Why time-based splits are essential
- Walk-forward validation methodology
- How to implement it in TensorTrade
- Interpreting walk-forward results

---

## The Problem with Random Splits

### Machine Learning Default

```python
# Standard ML: Random 80/20 split
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=True)
```

### Why It Fails for Time Series

```
Data:     [Jan] [Feb] [Mar] [Apr] [May] [Jun]

Random split might give:
  Train:  [Jan] [Mar] [May]  (scattered)
  Test:   [Feb] [Apr] [Jun]  (interleaved)

Problems:
  1. Agent sees Feb patterns when learning for Feb
  2. March data "leaks" information about February
  3. Test is not truly unseen future data
```

---

## Time-Based Split (Basic)

### The Correct Approach

```
Data:     [Jan] [Feb] [Mar] [Apr] [May] [Jun]

Time-based split:
  Train:  [Jan] [Feb] [Mar] [Apr]  (past)
  Test:   [May] [Jun]              (future)

Agent only sees past data during training
Test data is truly "unseen future"
```

### Implementation

```python
# Time-based split (no shuffling!)
total_candles = len(data)
test_size = 30 * 24   # 30 days
val_size = 30 * 24    # 30 days

test_data = data.iloc[-test_size:]
val_data = data.iloc[-(test_size + val_size):-test_size]
train_data = data.iloc[:-(test_size + val_size)]

# Timeline:
# [=============== train ===============][= val =][= test =]
#                                        ^        ^
#                                    val start  test start
```

---

## Walk-Forward Validation

Walk-forward simulates how you'd actually use the model:
1. Train on available data
2. Test on next period
3. Retrain with updated data
4. Repeat

```
┌────────────────────────────────────────────────────────────────┐
│                   Walk-Forward Validation                      │
│                                                                │
│  Fold 1: [=== TRAIN ===][TEST]                                │
│          Jan-Apr        May                                    │
│                                                                │
│  Fold 2:    [=== TRAIN ===][TEST]                             │
│             Feb-May       Jun                                  │
│                                                                │
│  Fold 3:       [=== TRAIN ===][TEST]                          │
│                Mar-Jun       Jul                               │
│                                                                │
│  Fold 4:          [=== TRAIN ===][TEST]                       │
│                   Apr-Jul       Aug                            │
│                                                                │
│  Final Result: Average of all fold test results               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Why Walk-Forward is Better

1. **Tests on multiple periods** - Not just one lucky/unlucky month
2. **Simulates real deployment** - Train → deploy → retrain cycle
3. **Reveals distribution shift** - How performance varies over time
4. **More robust results** - Average is more reliable than single test

---

## Implementation

### Basic Walk-Forward

```python
def walk_forward(data, train_months=4, test_months=1, stride_months=1):
    """Walk-forward validation for trading."""
    results = []
    candles_per_month = 30 * 24  # Hourly data

    train_size = train_months * candles_per_month
    test_size = test_months * candles_per_month
    stride = stride_months * candles_per_month

    # Calculate number of folds
    total_candles = len(data)
    start = 0

    fold = 0
    while start + train_size + test_size <= total_candles:
        fold += 1

        # Extract train and test
        train_end = start + train_size
        test_end = train_end + test_size

        train_data = data.iloc[start:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()

        print(f"\nFold {fold}:")
        print(f"  Train: {train_data['date'].iloc[0]} to {train_data['date'].iloc[-1]}")
        print(f"  Test:  {test_data['date'].iloc[0]} to {test_data['date'].iloc[-1]}")

        # Train model
        model = train(train_data)

        # Evaluate on test
        test_pnl = evaluate(model, test_data)
        bh_pnl = buy_and_hold(test_data)

        results.append({
            'fold': fold,
            'test_start': test_data['date'].iloc[0],
            'test_end': test_data['date'].iloc[-1],
            'agent_pnl': test_pnl,
            'bh_pnl': bh_pnl,
        })

        print(f"  Agent P&L: ${test_pnl:+,.0f}, B&H: ${bh_pnl:+,.0f}")

        # Move window forward
        start += stride

    return pd.DataFrame(results)
```

### Full Example

```python
def run_walk_forward():
    # Load data
    data = load_data()

    # Run walk-forward
    results = walk_forward(
        data,
        train_months=4,   # 4 months training
        test_months=1,    # 1 month test
        stride_months=1   # Move forward 1 month each fold
    )

    # Summary statistics
    print("\n" + "="*50)
    print("Walk-Forward Results Summary")
    print("="*50)

    avg_agent = results['agent_pnl'].mean()
    avg_bh = results['bh_pnl'].mean()
    win_rate = (results['agent_pnl'] > results['bh_pnl']).mean()

    print(f"Average Agent P&L: ${avg_agent:+,.0f}")
    print(f"Average B&H P&L:   ${avg_bh:+,.0f}")
    print(f"Win Rate vs B&H:   {win_rate:.1%}")
    print(f"Folds where Agent beats B&H: {sum(results['agent_pnl'] > results['bh_pnl'])}/{len(results)}")

    return results
```

---

## Interpreting Results

### Good Results

```
Walk-Forward Results Summary
============================
Fold 1: Agent $+150, B&H $-100  (Win)
Fold 2: Agent $+80,  B&H $+200  (Loss)
Fold 3: Agent $+210, B&H $+50   (Win)
Fold 4: Agent $-50,  B&H $-300  (Win)
Fold 5: Agent $+120, B&H $+100  (Win)

Average Agent: $+102
Average B&H:   $-10
Win Rate: 80% (4/5 folds)

Interpretation: Agent consistently beats B&H across market conditions
```

### Concerning Results

```
Walk-Forward Results Summary
============================
Fold 1: Agent $+500, B&H $-200  (Win)   ← Bull market, agent wins big
Fold 2: Agent $-400, B&H $-100  (Loss)  ← Bear market, agent loses more
Fold 3: Agent $+300, B&H $+250  (Win)   ← Bull market, agent wins small
Fold 4: Agent $-600, B&H $-300  (Loss)  ← Bear market, agent loses big
Fold 5: Agent $+100, B&H $+50   (Win)

Average Agent: $-20
Average B&H:   $-60
Win Rate: 60%

Interpretation: Agent does well in bull markets but badly in bear markets
  → Not robust, probably overfit to bullish patterns
```

### Red Flags

```
Warning signs in walk-forward results:

1. High variance across folds
   Fold 1: +$500, Fold 2: -$800, Fold 3: +$600
   → Strategy is unstable

2. Performance degradation over time
   Fold 1: +$300, Fold 2: +$200, Fold 3: +$50, Fold 4: -$100
   → Market regime changed, model didn't adapt

3. Wins only in specific market conditions
   All wins during rising markets, all losses during falling
   → Not actually predicting, just biased long/short
```

---

## Anchored vs Rolling Walk-Forward

### Rolling (Standard)

Training window moves with test window:

```
Fold 1: [Jan-Apr] → [May]
Fold 2: [Feb-May] → [Jun]
Fold 3: [Mar-Jun] → [Jul]

Each fold uses same amount of training data
Older data "falls off" the training set
```

### Anchored (Expanding)

Training window grows over time:

```
Fold 1: [Jan-Apr]     → [May]
Fold 2: [Jan-May]     → [Jun]
Fold 3: [Jan-Jun]     → [Jul]

Each fold uses more training data
Never discards older data
```

### Which to Use?

| Method | Pros | Cons |
|--------|------|------|
| **Rolling** | Adapts to recent conditions | Loses old patterns |
| **Anchored** | More training data | May include stale patterns |

**Recommendation**: Start with rolling. Try anchored if you have limited data.

---

## Walk-Forward with Retraining

In production, you'd retrain periodically:

```python
def walk_forward_with_retraining(data):
    """Walk-forward with model retraining."""
    results = []
    model = None

    for fold in range(n_folds):
        train_data = get_train_data(fold)
        test_data = get_test_data(fold)

        # Retrain from scratch each fold
        # (Or fine-tune from previous model)
        model = train(train_data)

        test_pnl = evaluate(model, test_data)
        results.append(test_pnl)

    return results
```

### Fine-Tuning vs Fresh Training

```python
# Option A: Train from scratch each fold
model = PPOConfig().build()  # Fresh model

# Option B: Fine-tune from previous fold
if previous_model:
    model = previous_model
    model.train()  # Continue training with new data
else:
    model = PPOConfig().build()
```

Fine-tuning is faster but may accumulate errors. Fresh training is slower but more robust.

---

## Statistical Significance

### The Problem

```
5 folds: Agent wins 3, loses 2

Is this skill or luck?
- Could be 60% real win rate (skill)
- Could be 50% and we got lucky (noise)
```

### Simple Significance Test

```python
from scipy import stats

def test_significance(agent_pnls, bh_pnls):
    """Test if agent significantly beats B&H."""
    differences = [a - b for a, b in zip(agent_pnls, bh_pnls)]

    # One-sample t-test: Is mean difference > 0?
    t_stat, p_value = stats.ttest_1samp(differences, 0)

    print(f"Mean difference: ${np.mean(differences):+,.0f}")
    print(f"t-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05 and np.mean(differences) > 0:
        print("Result: Statistically significant outperformance")
    else:
        print("Result: Cannot conclude agent beats B&H")
```

### Sample Size Requirements

```
For reliable results:
- Minimum 5-10 folds
- Preferably 20+ folds
- More folds = more confident results

With 5 folds:
  Even 5/5 wins might not be significant
  Could still be luck

With 20 folds:
  15/20 wins is more convincing
  Harder to achieve by chance
```

---

## Common Mistakes

### Mistake 1: Using Future Data in Features

```python
# WRONG: Using .shift(-1) includes future
df['future_return'] = df['close'].shift(-1) / df['close'] - 1
```

### Mistake 2: Overlapping Train/Test

```python
# WRONG: Validation overlaps with training
train_data = data.iloc[:1000]
val_data = data.iloc[900:1100]  # 100 rows overlap!
```

### Mistake 3: Testing on Same Period Multiple Times

```python
# WRONG: Tuning hyperparameters on test set
for lr in [0.001, 0.0001, 0.00001]:
    model = train(train_data, lr=lr)
    test_pnl = evaluate(model, test_data)  # Peeking at test!

# Now test_pnl is optimistic because you tuned on it
```

### Mistake 4: Ignoring Market Regimes

```python
# WRONG: Not checking if results are regime-dependent
# Agent might only work in certain market conditions
```

---

## Key Takeaways

1. **Never shuffle time series data** - Use time-based splits
2. **Walk-forward simulates real deployment** - Train → test → retrain
3. **Multiple folds give robust results** - Single test is unreliable
4. **Check for regime dependency** - Does it work in all market conditions?
5. **Statistical significance matters** - 3/5 wins might be luck

---

## Checkpoint

After this tutorial, verify you understand:

- [ ] Why random splits fail for time series
- [ ] How walk-forward validation works
- [ ] The difference between rolling and anchored
- [ ] How to interpret walk-forward results

---

## Final Words

You've completed the TensorTrade tutorial curriculum.

**What we learned:**
- RL agents CAN predict market direction
- Commission is the main challenge
- Overfitting is the default failure mode
- Proper validation is essential

**What's next:**
- Contribute to TensorTrade (reduce overtrading!)
- Experiment with your own strategies
- Join the community on Discord

Happy trading!
