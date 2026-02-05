# First Training

This tutorial walks you through training a real RL agent with Ray RLlib.

## Learning Objectives

After this tutorial, you will:
- Successfully train a PPO agent
- Understand training output and metrics
- Know how to evaluate your agent
- Recognize signs of overfitting

---

## Prerequisites

```bash
# Install training dependencies
pip install -r examples/requirements.txt
```

This installs:
- Ray RLlib (distributed training)
- PyTorch (neural networks)
- Optuna (hyperparameter tuning)

---

## The Training Script

We'll use `train_best.py` which implements our best configuration from experiments.

```bash
python examples/training/train_best.py
```

### What It Does

1. **Loads data** - Historical BTC/USD prices
2. **Adds features** - Scale-invariant indicators
3. **Splits data** - Train / Validation / Test
4. **Trains PPO** - With best hyperparameters
5. **Evaluates** - On held-out test data

---

## Understanding the Code

### 1. Data Preparation

```python
# Fetch data
cdd = CryptoDataDownload()
data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")

# Add scale-invariant features
data = add_features(data)
feature_cols = [c for c in data.columns
                if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]

# Split data (time-based, no shuffling!)
test_candles = 30 * 24      # Last 30 days
val_candles = 30 * 24       # Previous 30 days
train_data = data.iloc[:-(test_candles + val_candles)].tail(4000)
val_data = data.iloc[-(test_candles + val_candles):-test_candles]
test_data = data.iloc[-test_candles:]
```

**Key points:**
- Time-based split (never shuffle time series!)
- 4000 candles for training (~167 days)
- 720 candles each for validation and test (~30 days each)

### 2. Environment Configuration

```python
env_config = {
    "csv_filename": train_csv,
    "feature_cols": feature_cols,
    "window_size": 17,
    "max_allowed_loss": 0.32,
    "commission": 0.003,  # 0.3% for training
    "initial_cash": 10000,
}
```

**Key settings:**
- `window_size=17` - Agent sees 17 hours of history
- `max_allowed_loss=0.32` - Episode ends at 32% loss
- `commission=0.003` - Higher commission during training teaches discipline

### 3. PPO Configuration

```python
config = (
    PPOConfig()
    .environment(env="TradingEnv", env_config=env_config)
    .framework("torch")
    .env_runners(num_env_runners=4)
    .training(
        lr=3.29e-05,           # Very low learning rate
        gamma=0.992,           # High discount factor
        lambda_=0.9,           # GAE parameter
        clip_param=0.123,      # PPO clipping
        entropy_coeff=0.015,   # Low entropy (exploitation)
        train_batch_size=2000,
        sgd_minibatch_size=256,
        num_sgd_iter=7,
        vf_clip_param=100.0,
        model={"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"},
    )
)
```

**Why these values?** Found via Optuna optimization over 100 trials.

### 4. Training Loop

```python
algo = config.build()

for i in range(100):
    result = algo.train()

    if (i + 1) % 10 == 0:
        # Evaluate on validation set
        val_pnl = evaluate(algo, val_data, feature_cols, config, n=10)

        # Track best model
        if val_pnl > best_val:
            best_val = val_pnl
            algo.save('/tmp/best_model')
            marker = " *"

        print(f"Iter {i+1}: Train ${pnl:+,.0f} | Val ${val_pnl:+,.0f}{marker}")
```

**What happens:**
1. Train for 100 iterations
2. Every 10 iterations, evaluate on validation set
3. Save model when validation improves
4. Use best validation model for testing

---

## Reading the Output

### Training Output

```
======================================================================
Training (100 iterations, zero commission)
======================================================================
  Iter  10: Train $+1,234 | Val $-456
  Iter  20: Train $+2,567 | Val $-234  *
  Iter  30: Train $+3,891 | Val $-189  *
  Iter  40: Train $+4,234 | Val $-312
  Iter  50: Train $+5,678 | Val $-289
```

**What to look for:**
- Training P&L increasing (agent learning)
- Validation P&L is the real metric
- `*` indicates new best validation
- When training ↑ but validation ↓ → overfitting!

### Test Results

```
======================================================================
Test Results
======================================================================

Test period: 2025-09-16 to 2025-10-16
BTC: $115,290 -> $111,200

========================================
Agent (0% commission):     $+239
Agent (0.1% commission):   $-650
Agent (0.2% commission):   $-1,200
Buy & Hold:                $-355
========================================

*** PROFITABLE at 0% commission! ***
```

**Interpretation:**
- At 0% commission: Agent made +$239 (direction prediction works!)
- At 0.1% commission: Agent lost -$650 (commission destroyed profit)
- Buy-and-Hold lost $355 (market went down)
- Agent beats B&H by $594 when commission is zero

---

## Key Metrics to Track

### During Training

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Training P&L | Increasing | Stuck or decreasing |
| Validation P&L | Improving | Declining while train improves |
| Episode reward | Higher over time | Flat or erratic |

### During Evaluation

| Metric | What It Tells You |
|--------|-------------------|
| P&L (0% commission) | Pure direction prediction ability |
| P&L (0.1% commission) | Realistic performance |
| Number of trades | Trading frequency |
| Trades per day | Should be 1-5, not 100+ |

---

## Detecting Overfitting

### Visual Pattern

```
Training vs Validation P&L Over Iterations:

        │
  P&L   │         Training ──────────────────▶
        │              ╱
        │            ╱
        │          ╱           Validation
        │        ╱              ────────────╮
        │      ╱                             ╲
        │    ╱                                ╲
        │──╱───────────────────────────────────╲───▶
        │                                        Iteration
        │
        │  ▲
        │  │ OVERFITTING starts here
        │  │ (train improving, val declining)
```

### In Numbers

```
Iter  50: Train $+3,000 | Val $-100  *  (learning)
Iter  60: Train $+4,000 | Val $-150      (still ok)
Iter  70: Train $+5,500 | Val $-300      (warning!)
Iter  80: Train $+7,000 | Val $-600      (overfitting!)
Iter  90: Train $+8,500 | Val $-900      (stop here!)
```

### What to Do

1. **Early stopping** - Use the model from iteration 60, not 90
2. **Reduce model size** - Try `[64, 64]` instead of `[128, 128]`
3. **Increase regularization** - Higher entropy coefficient
4. **Add more data** - More training candles

---

## Practical Exercise

### Exercise 1: Run Training

```bash
python examples/training/train_best.py
```

Expected runtime: ~10-20 minutes (depends on hardware)

### Exercise 2: Modify Commission

Edit `train_best.py`:

```python
# Try different training commission
env_config = {
    ...
    "commission": 0.001,  # 0.1% instead of 0.3%
}
```

Run again. Does the agent trade more? Less?

### Exercise 3: Change Network Size

```python
config = (
    PPOConfig()
    ...
    .training(
        ...
        model={"fcnet_hiddens": [64, 64]},  # Smaller network
    )
)
```

Does it overfit less?

---

## Understanding Results

### Scenario: Validation Beats Training

```
Iter 100: Train $-500 | Val $+200
```

Unusual but possible if:
- Validation period had clearer trends
- Agent learned conservative strategy that works better on val

### Scenario: Both Negative

```
Iter 100: Train $-1,500 | Val $-800
```

Agent hasn't learned useful patterns yet:
- Train longer (more iterations)
- Check features (are they informative?)
- Try different hyperparameters

### Scenario: Good Val, Bad Test

```
Val: $+500 | Test: $-1,200
```

Agent overfit to validation set:
- Don't tune hyperparameters too much on val
- Market conditions changed between val and test
- This is common - financial markets are hard

---

## Tips for Better Training

### 1. Start Simple

```python
# Begin with fewer features
feature_cols = ['ret_1h', 'ret_24h', 'rsi', 'trend']  # Just 4

# And smaller network
model={"fcnet_hiddens": [32, 32]}

# Get something working first, then iterate
```

### 2. Watch the First 20 Iterations

```python
# If validation is improving in first 20 iterations
# Your setup is probably correct

# If flat or random in first 20 iterations
# Check: features, reward, commission level
```

### 3. Use More Workers for Faster Training

```python
.env_runners(num_env_runners=8)  # Use 8 parallel environments
```

Requires more CPU cores but trains faster.

### 4. Log More Metrics

```python
# In callback
episode.custom_metrics["trades"] = count_trades(env)
episode.custom_metrics["holding_pct"] = holding_time / total_time
```

Track more than just P&L.

---

## Troubleshooting

### "ValueError: observation shape mismatch"

Features don't match between train and eval:
```python
# Make sure same feature_cols used everywhere
print(f"Features: {len(feature_cols)}")
print(feature_cols)
```

### "NaN reward"

Features contain NaN:
```python
data = add_features(data).bfill().ffill()
assert not data[feature_cols].isna().any().any()
```

### "Episode ended immediately"

Max loss triggered at start:
```python
# Increase max_allowed_loss
"max_allowed_loss": 0.5  # 50% instead of 32%
```

### Training takes forever

Reduce batch size or workers:
```python
train_batch_size=1000,  # Smaller batch
num_env_runners=2,      # Fewer workers
```

---

## Key Takeaways

1. **Use train_best.py** - It has the best configuration from experiments
2. **Watch validation P&L** - That's the real performance metric
3. **Stop when validation stops improving** - Continuing leads to overfitting
4. **0% commission test shows direction prediction** - The core ability
5. **Commission destroys profit** - The unsolved challenge

---

## Checkpoint

After training, verify:

- [ ] Training completed without errors
- [ ] Validation P&L improved during training
- [ ] You can identify where overfitting starts
- [ ] Test P&L shows agent can predict direction (at 0% commission)

---

## Next Steps

[02-ray-rllib.md](02-ray-rllib.md) - Deep dive into Ray RLlib configuration
