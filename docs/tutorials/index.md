# TensorTrade Tutorials

Welcome to TensorTrade! This curriculum will teach you to build RL trading agents.

---

## The Discovery

We trained RL agents to trade BTC/USD and discovered:

| Experiment | Test P&L | vs Buy-and-Hold |
|------------|----------|-----------------|
| Agent (0% commission) | **+$239** | **+$594** |
| Agent (0.1% commission) | -$650 | -$295 |

**The agent CAN predict direction.** The challenge is overtrading.

---

## Learning Paths

### Path 1: Quick Start (30 minutes)

Get something working fast:

1. [Three Pillars](01-foundations/01-three-pillars.md) - Understand the domains
2. [Your First Run](01-foundations/03-your-first-run.md) - Run `train_simple.py`
3. [First Training](04-training/01-first-training.md) - Train a real agent

### Path 2: Full Curriculum (1-2 days)

Comprehensive understanding:

```
Module 1: Foundations
├── 01-three-pillars.md      # RL + Trading + Data
├── 02-architecture.md       # How components work
└── 03-your-first-run.md     # Run and understand output

Module 2: Domain Knowledge (choose your track)
├── Track A: Trading for RL People
│   ├── 01-trading-basics.md
│   └── 02-oms-deep-dive.md
├── Track B: RL for Traders
│   ├── 01-rl-fundamentals.md
│   └── 02-common-failures.md  ← CRITICAL
└── Track C: Full Introduction
    └── README.md

Module 3: Core Components
├── 01-action-schemes.md     # BSH explained
├── 02-reward-schemes.md     # Why PBR works
└── 03-observers-feeds.md    # Feature engineering

Module 4: Training
├── 01-first-training.md     # Train with Ray RLlib
├── 02-ray-rllib.md          # Configuration deep dive
└── 03-optuna.md             # Hyperparameter optimization

Module 5: Advanced
├── 01-overfitting.md        # Detection and prevention
├── 02-commission.md         # THE breakthrough finding
└── 03-walk-forward.md       # Proper validation
```

---

## Where to Start?

### Coming from RL?
Start with [Trading Basics](02-domains/track-a-trading-for-rl/01-trading-basics.md)

### Coming from Trading/Quant?
Start with [RL Fundamentals](02-domains/track-b-rl-for-traders/01-rl-fundamentals.md)

### New to Both?
Start with [Full Introduction](02-domains/track-c-full-intro/README.md)

### Want to Jump In?
Go directly to [First Training](04-training/01-first-training.md)

---

## Critical Reading

Before you invest serious time, read these:

1. **[Common Failures](02-domains/track-b-rl-for-traders/02-common-failures.md)** - What destroys RL trading agents
2. **[Commission Analysis](05-advanced/02-commission.md)** - Our breakthrough discovery
3. **[Overfitting](05-advanced/01-overfitting.md)** - The default failure mode

---

## Quick Reference

### Training Scripts

| Script | Purpose |
|--------|---------|
| `train_simple.py` | Demo with wallet balances |
| `train_ray_long.py` | Distributed training |
| `train_optuna.py` | Hyperparameter optimization |
| `train_best.py` | Best configuration |

### Key Components

| Component | Default | Purpose |
|-----------|---------|---------|
| ActionScheme | BSH | Convert actions to trades |
| RewardScheme | PBR | Learning signal |
| Observer | TensorTrade | Create observations |

### Best Hyperparameters

```python
{
    "lr": 3.29e-05,
    "gamma": 0.992,
    "entropy": 0.015,
    "clip": 0.123,
    "hidden": [128, 128],
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TradingEnv                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Episode Loop                           │  │
│  │                                                           │  │
│  │   ┌─────────┐    ┌──────────┐    ┌────────────────┐     │  │
│  │   │Observer │───>│  Agent   │───>│  ActionScheme  │     │  │
│  │   │(features)    │(RL model)│    │  (BSH/Orders)  │     │  │
│  │   └─────────┘    └──────────┘    └───────┬────────┘     │  │
│  │        ^                                  │              │  │
│  │        │         ┌──────────┐            v              │  │
│  │        │         │ Reward   │<───── Portfolio           │  │
│  │        │         │ Scheme   │       (Wallets)           │  │
│  │        │         │ (PBR)    │                           │  │
│  │        │         └────┬─────┘                           │  │
│  │        └──────────────┘                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

After completing this curriculum, you should be able to:

| Level | Time | Capability |
|-------|------|------------|
| 1 | 5 min | Run code and see a trading agent |
| 2 | 30 min | Understand the core architecture |
| 3 | 2 hours | Modify components and run experiments |
| 4 | 1 day | Train a real agent and understand results |
| 5 | 1 week | Build custom components, avoid pitfalls |

---

## Additional Resources

- **[EXPERIMENTS.md](../EXPERIMENTS.md)** - Full research log
- **[API Documentation](https://tensortrade.org)** - Reference docs
- **[Discord](https://discord.gg/ZZ7BGWh)** - Community support

---

## Contributing

TensorTrade needs help with:

1. **Reduce overtrading** - The agent trades too frequently
2. **Position sizing** - Replace binary BSH with continuous actions
3. **Commission-aware rewards** - Include fees in learning signal

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

Happy trading!
