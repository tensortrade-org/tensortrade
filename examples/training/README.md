# TensorTrade Training Scripts

This directory contains training scripts for reinforcement learning agents using TensorTrade with Ray RLlib.

## Available Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `train_simple.py` | Basic local training demo | Getting started, quick testing |
| `train_ray_long.py` | Distributed PPO training | Production training with Ray |
| `train_optuna.py` | Hyperparameter optimization | Finding optimal parameters |
| `train_best.py` | Best known configuration | Recommended starting point |
| `train_advanced.py` | AdvancedPBR reward scheme | Experimenting with trade penalties |
| `train_profit.py` | Sharpe ratio + bear markets | Alternative reward schemes |
| `train_walkforward.py` | Walk-forward validation | Multi-regime training |
| `train_robust.py` | Anti-overfitting approach | Scale-invariant features |
| `train_trend.py` | Minimal trend-following | Simple 5-feature model |
| `train_historical.py` | Historical data training | Custom date ranges |
| `run_ray_simulation.py` | Distributed simulation runner | Parallel evaluation |

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run simple training demo
python train_simple.py

# Run distributed training with Ray
python train_ray_long.py

# Run hyperparameter optimization
python train_optuna.py
```

## Recommended Workflow

1. **Start with `train_best.py`** - Uses optimized hyperparameters from 100+ Optuna trials
2. **Check zero-commission performance** - Measures pure direction prediction ability
3. **Tune training commission** - 0.3-0.5% seems optimal for learning trading discipline
4. **Compare to Buy-and-Hold** - Always report performance vs B&H baseline

## Key Findings

See [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md) for detailed experiment results.

**Summary:**
- Agent CAN predict market direction profitably (+$239 profit with zero commission)
- The challenge is **overtrading** - commission costs wipe out profits
- Training with higher commission (0.3-0.5%) improves trading discipline
- Best hyperparameters: low learning rate (3e-5), tight clipping (0.05-0.1), high gamma (0.99+)

## Requirements

```bash
pip install ray[default,tune,rllib,serve]==2.37.0
pip install optuna
```

Or install all example dependencies:
```bash
pip install -r ../requirements.txt
```
