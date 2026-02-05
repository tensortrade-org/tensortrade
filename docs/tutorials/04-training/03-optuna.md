# Hyperparameter Optimization with Optuna

Manually tuning hyperparameters is tedious. Optuna automates the search.

## Learning Objectives

After this tutorial, you will understand:
- What Optuna does and how it works
- Which hyperparameters to optimize
- How to set up an optimization study
- How to interpret results

---

## What is Optuna?

Optuna is a hyperparameter optimization framework that:
- Samples hyperparameter combinations intelligently (not random)
- Prunes unpromising trials early (saves time)
- Tracks all trials and results
- Finds good hyperparameters automatically

```
Traditional approach:
  lr=0.001 → train → evaluate → try again
  lr=0.0001 → train → evaluate → try again
  lr=0.00001 → train → evaluate → ...
  (manual, slow, incomplete search)

Optuna approach:
  Trial 1: lr=0.0003 → P&L $-500
  Trial 2: lr=0.00008 → P&L $-300 (better, explore this area)
  Trial 3: lr=0.00012 → P&L $-200 (even better!)
  Trial 4: lr=0.00015 → P&L $-400 (worse, try other direction)
  ...
  (automated, intelligent search)
```

---

## Setup

```bash
pip install optuna
```

---

## Basic Optuna Study

### The Objective Function

Optuna calls this function for each trial:

```python
import optuna

def objective(trial):
    # 1. Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    entropy = trial.suggest_float("entropy", 0.001, 0.1, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

    # 2. Configure and train
    config = (
        PPOConfig()
        .training(
            lr=lr,
            gamma=gamma,
            entropy_coeff=entropy,
            model={"fcnet_hiddens": [hidden_size, hidden_size]},
        )
        ...
    )

    algo = config.build()

    # Train for N iterations
    for i in range(50):
        algo.train()

    # 3. Evaluate on validation set
    val_pnl = evaluate(algo, val_data)

    algo.stop()

    # 4. Return metric to optimize (Optuna MAXIMIZES by default)
    return val_pnl  # Higher is better
```

### Running the Study

```python
# Create study
study = optuna.create_study(
    direction="maximize",  # Maximize validation P&L
    sampler=optuna.samplers.TPESampler(),  # Smart sampling
    pruner=optuna.pruners.MedianPruner()   # Early stopping
)

# Run optimization
study.optimize(objective, n_trials=20)

# Best results
print(f"Best P&L: ${study.best_value:+,.0f}")
print(f"Best params: {study.best_params}")
```

---

## Our Optuna Setup

From `train_optuna.py`:

### Search Space

```python
def objective(trial):
    # Learning rate (log scale, spans 2 orders of magnitude)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Discount factor (high values for long-term thinking)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)

    # Entropy (exploration vs exploitation)
    entropy = trial.suggest_float("entropy", 0.001, 0.1, log=True)

    # PPO clipping (prevents large policy changes)
    clip = trial.suggest_float("clip", 0.01, 0.3)

    # Network architecture
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

    # Observation window
    window_size = trial.suggest_int("window_size", 5, 30)

    # Training parameters
    sgd_iters = trial.suggest_int("sgd_iters", 3, 15)
    batch_size = trial.suggest_categorical("batch_size",
                                           [1000, 2000, 4000, 8000])

    # Trading parameters
    commission = trial.suggest_float("commission", 0.0001, 0.005)
    max_loss = trial.suggest_float("max_loss", 0.1, 0.5)
```

### Training with Pruning

```python
def objective(trial):
    # ... sample params ...

    # Configure algorithm
    config = (
        PPOConfig()
        .training(lr=lr, gamma=gamma, ...)
    )
    algo = config.build()

    # Train with intermediate reporting
    for i in range(50):
        result = algo.train()

        # Report intermediate value
        if (i + 1) % 10 == 0:
            val_pnl = evaluate(algo, val_data)
            trial.report(val_pnl, i)

            # Optuna may decide to prune this trial
            if trial.should_prune():
                algo.stop()
                raise optuna.TrialPruned()

    # Final evaluation
    val_pnl = evaluate(algo, val_data)
    algo.stop()
    return val_pnl
```

---

## Results from 100 Trials

### Best Hyperparameters Found

```python
{
    "lr": 3.29e-05,           # Very low learning rate
    "entropy": 0.015,         # Low entropy (exploitation)
    "gamma": 0.992,           # High discount factor
    "clip": 0.123,            # Moderate clipping
    "hidden_size": 128,       # Larger network
    "window_size": 17,        # ~17 hours of history
    "max_loss": 0.32,         # 32% stop loss
    "commission": 0.00013,    # Very low for training
    "sgd_iters": 7,           # Multiple SGD passes
    "batch_size": 2000        # Moderate batch
}
```

### Top 5 Trials

| Trial | Val P&L | lr | entropy | hidden |
|-------|---------|-----|---------|--------|
| 48 | -$125 | 3.29e-05 | 0.015 | 128 |
| 97 | -$189 | 6.70e-05 | 0.093 | 32 |
| 70 | -$245 | 4.20e-05 | 0.039 | 64 |
| 81 | -$272 | 6.63e-05 | 0.068 | 32 |
| 89 | -$300 | 8.38e-05 | 0.077 | 32 |

### What We Learned

1. **Very low learning rates work best** - 3e-5 to 8e-5
2. **Low entropy is better** - Agent should exploit more
3. **Moderate network size** - 128 neurons won, but 32 also good
4. **High gamma** - Value future rewards

---

## Visualization

Optuna provides built-in visualizations:

```python
import optuna.visualization as vis

# Parameter importances
fig = vis.plot_param_importances(study)
fig.show()

# Optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Contour plot for 2 parameters
fig = vis.plot_contour(study, params=["lr", "entropy"])
fig.show()
```

### Sample Output

```
Parameter Importances:
  lr: 0.35           ████████████████████
  entropy: 0.22      ████████████
  gamma: 0.18        ██████████
  hidden_size: 0.12  ███████
  window_size: 0.08  █████
  clip: 0.05         ███
```

Learning rate is the most important parameter!

---

## Tips for Effective Optimization

### 1. Start with Fewer Trials

```python
# Quick exploration
study.optimize(objective, n_trials=10)  # 10 trials first

# If results look promising, continue
study.optimize(objective, n_trials=40)  # 40 more trials
```

### 2. Use Reasonable Bounds

```python
# BAD: Too wide
lr = trial.suggest_float("lr", 1e-8, 1e-1, log=True)

# GOOD: Focused range
lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
```

### 3. Enable Pruning

```python
# Prunes bad trials early (saves time)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,    # Don't prune first 5 trials
    n_warmup_steps=10,     # Don't prune first 10 reports
)

study = optuna.create_study(pruner=pruner)
```

### 4. Save Study Progress

```python
# Save to SQLite (can resume later)
study = optuna.create_study(
    storage="sqlite:///optuna_study.db",
    study_name="trading_optimization",
    load_if_exists=True,
)

# Can resume after crash or manual stop
```

### 5. Use Multiple Objectives (Pareto)

```python
# Optimize both P&L and trade count
study = optuna.create_study(
    directions=["maximize", "minimize"],  # Max P&L, min trades
)

def objective(trial):
    # ... train ...
    return val_pnl, num_trades  # Return both
```

---

## Full Example Script

```python
#!/usr/bin/env python3
"""Optuna hyperparameter optimization for TensorTrade."""

import optuna
import ray
from ray.rllib.algorithms.ppo import PPOConfig

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    entropy = trial.suggest_float("entropy", 0.001, 0.1, log=True)
    clip = trial.suggest_float("clip", 0.01, 0.3)
    hidden = trial.suggest_categorical("hidden_size", [32, 64, 128])

    # Build configuration
    config = (
        PPOConfig()
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .training(
            lr=lr,
            gamma=gamma,
            entropy_coeff=entropy,
            clip_param=clip,
            model={"fcnet_hiddens": [hidden, hidden]},
        )
    )

    algo = config.build()

    # Train with pruning
    for i in range(50):
        algo.train()

        if (i + 1) % 10 == 0:
            val_pnl = evaluate(algo, val_data)
            trial.report(val_pnl, i)

            if trial.should_prune():
                algo.stop()
                raise optuna.TrialPruned()

    val_pnl = evaluate(algo, val_data)
    algo.stop()
    return val_pnl

if __name__ == "__main__":
    ray.init(num_cpus=6, ignore_reinit_error=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(objective, n_trials=20)

    print(f"\nBest P&L: ${study.best_value:+,.0f}")
    print(f"Best params: {study.best_params}")

    ray.shutdown()
```

---

## When to Stop Optimizing

### Diminishing Returns

```
Trials 1-10:   Best P&L improves from -$500 to -$300
Trials 10-20:  Best P&L improves from -$300 to -$200
Trials 20-50:  Best P&L improves from -$200 to -$150
Trials 50-100: Best P&L improves from -$150 to -$125

After 100 trials, improvements are minimal.
Further optimization unlikely to help much.
```

### Validation-Test Gap

```
If best validation P&L is -$125
But test P&L is -$650

The hyperparameters overfit to validation.
More trials won't help - need different approach.
```

---

## Key Takeaways

1. **Optuna automates hyperparameter search** - Much better than manual tuning
2. **Use log scale for learning rate** - Spans multiple orders of magnitude
3. **Enable pruning** - Stops bad trials early
4. **Learning rate is most important** - Focus search there
5. **Diminishing returns after ~50-100 trials** - Don't over-optimize

---

## Checkpoint

After running Optuna, verify:

- [ ] Study completed without errors
- [ ] Best parameters are printed
- [ ] You understand which parameters matter most
- [ ] You know when to stop optimizing

---

## Next Steps

Advanced topics:
- [Overfitting](../05-advanced/01-overfitting.md)
- [Commission Analysis](../05-advanced/02-commission.md)
