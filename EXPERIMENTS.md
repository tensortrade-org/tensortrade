# TensorTrade Training Experiments Log

This document summarizes the RL training experiments conducted to optimize trading performance.

## Overview

**Goal:** Train a PPO agent to trade BTC/USD profitably using the TensorTrade framework with Ray RLlib.

**Test Period:** September 16 - October 16, 2025 (30 days)
**Market Conditions:** BTC dropped from $115,290 to $111,200 (-3.55%)
**Benchmark:** Buy-and-Hold lost $355 (-3.55%)

---

## Experiments Summary

| # | Approach | Features | Network | Test P&L | vs B&H |
|---|----------|----------|---------|----------|--------|
| 1 | Walk-forward | 34 | 256x256 | -$2,690 | -$2,335 |
| 2 | Scale-invariant | 22 | 64x64 | -$2,496 | -$2,141 |
| 3 | Trend-following | 5 | 32x32 | -$1,680 | -$1,325 |
| 4 | **Optuna optimized** | 13 | 64x64 | **-$748** | **-$393** |

---

## Experiment 1: Walk-Forward Training

**Script:** `train_walkforward.py`

**Approach:**
- Combined data from multiple market regimes (bull, bear, sideways)
- 34 technical indicators
- 150 training iterations

**Results:**
- Training P&L improved from -$6,005 to +$6,366
- Test P&L: -$2,690 (lost 7.6x more than B&H)

**Conclusion:** Model overfits to training patterns that don't transfer.

---

## Experiment 2: Anti-Overfitting (Scale-Invariant)

**Script:** `train_robust.py`

**Approach:**
- Only scale-invariant features (percentages, ratios)
- Smaller network (64x64)
- High entropy (0.05) for exploration
- Early stopping on validation set

**Results:**
- Early stopped at iteration 50
- Test P&L: -$2,496

**Conclusion:** Modest improvement, but still significant overfitting.

---

## Experiment 3: Minimal Trend-Following

**Script:** `train_trend.py`

**Approach:**
- Only 5 trend features:
  - `trend`: Price vs 50-period SMA
  - `momentum`: 24-hour return
  - `rsi_norm`: RSI normalized to [-1, 1]
  - `vol_regime`: High/low volatility
  - `trend_strength`: SMA10 vs SMA30
- Tiny network (32x32)
- Very high entropy (0.1)
- Lower commission (0.05%)

**Results:**
- Test P&L: -$1,680
- 38% improvement over baseline

**Conclusion:** Simpler models generalize better.

---

## Experiment 4: Optuna Hyperparameter Optimization

**Script:** `train_optuna.py`

**Approach:**
- 20 trials with TPE sampler
- Median pruning for underperforming trials
- Optimized 10 hyperparameters

**Best Hyperparameters Found:**
```python
{
    "lr": 0.00014,           # Low learning rate
    "entropy": 0.053,        # Moderate exploration
    "gamma": 0.9975,         # High discount (values future rewards)
    "clip": 0.054,           # Very tight clipping
    "hidden_size": 64,
    "window_size": 19,
    "max_loss": 0.205,       # 20% stop loss
    "commission": 0.0001,    # 0.01% assumed
    "sgd_iters": 11,
    "batch_size": 8000
}
```

**Results:**
- Best validation P&L: -$447
- Test P&L: -$748
- **72% improvement** over baseline

**Top 5 Trials:**
| Trial | Val P&L | lr | entropy | hidden |
|-------|---------|-----|---------|--------|
| 10 | -$447 | 1.42e-04 | 0.053 | 64 |
| 2 | -$452 | 1.53e-04 | 0.011 | 128 |
| 11 | -$519 | 1.75e-04 | 0.053 | 64 |
| 13 | -$565 | 8.21e-05 | 0.078 | 64 |
| 19 | -$640 | 1.14e-04 | 0.024 | 32 |

---

## Key Learnings

### What Worked
1. **Lower learning rates** (0.0001-0.0002) - More stable learning
2. **Tight clipping** (0.05-0.1) - Prevents large policy changes
3. **High gamma** (0.99+) - Values long-term rewards
4. **Fewer features** - Reduces overfitting
5. **Early stopping** - Prevents overtraining

### What Didn't Work
1. **Many features** - 34 indicators led to overfitting
2. **Large networks** - 256x256 memorized training data
3. **High learning rates** - Unstable training
4. **Training too long** - Reward improved but generalization decreased

### The Fundamental Challenge

Even with optimization, the agent underperforms buy-and-hold. This is because:

1. **Distribution shift** - Market dynamics change over time
2. **Commission costs** - Active trading accumulates fees
3. **Pattern non-stationarity** - What worked before may not work again

This is realistic for algorithmic trading - beating the market consistently is extremely difficult.

---

## Running the Experiments

### Quick Start
```bash
# Simple local training
python train_simple.py

# Ray RLlib with wallet tracking
python train_ray_long.py

# Optuna optimization (recommended)
python train_optuna.py
```

### Requirements
```bash
pip install -r examples/requirements.txt
pip install optuna
```

### Customization

Edit the scripts to modify:
- **Data period**: Change `tail(N)` for different training lengths
- **Features**: Add/remove indicators in `add_features()` function
- **Hyperparameters**: Modify the `PPOConfig` or Optuna search space
- **Trials**: Increase `n_trials` in Optuna for better optimization

---

## Future Improvements

1. **More Optuna trials** - 50-100 trials for better optimization
2. **Different algorithms** - Try DQN, A2C, SAC
3. **Alternative rewards** - Sharpe ratio, risk-adjusted returns
4. **Walk-forward validation** - Rolling train/test windows
5. **Ensemble methods** - Combine multiple models
6. **Online learning** - Continuous retraining on new data

---

## References

- [TensorTrade Documentation](https://www.tensortrade.org/)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/)
- [Optuna](https://optuna.org/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
