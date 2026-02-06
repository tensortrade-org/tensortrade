<p align="center">
  <img src="docs/source/_static/logo.jpg" width="200" alt="TensorTrade Logo">
</p>

# TensorTrade

**Train RL agents to trade. Can they beat Buy-and-Hold?**

[![Tests](https://github.com/tensortrade-org/tensortrade/actions/workflows/tests.yml/badge.svg)](https://github.com/tensortrade-org/tensortrade/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/tensortrade/badge/?version=latest)](https://tensortrade.org)
[![Apache License](https://img.shields.io/github/license/tensortrade-org/tensortrade.svg?color=brightgreen)](http://www.apache.org/licenses/LICENSE-2.0)
[![Discord](https://img.shields.io/discord/592446624882491402.svg?color=brightgreen)](https://discord.gg/ZZ7BGWh)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

TensorTrade is an open-source Python framework for building, training, and evaluating reinforcement learning agents for algorithmic trading. The framework provides composable components for environments, action schemes, reward functions, and data feeds that can be combined to create custom trading systems.

## Quick Start

```bash
# Requires Python 3.12+
python3.12 -m venv tensortrade-env && source tensortrade-env/bin/activate
pip install -e .

# For training with Ray/RLlib (recommended)
pip install -r examples/requirements.txt

# Run training
python examples/training/train_simple.py
```

## Documentation & Tutorials

📚 **[Tutorial Index](docs/tutorials/index.md)** — Start here for the complete learning curriculum.

### Foundations
- [The Three Pillars](docs/tutorials/01-foundations/01-three-pillars.md) — RL + Trading + Data concepts
- [Architecture](docs/tutorials/01-foundations/02-architecture.md) — How components work together
- [Your First Run](docs/tutorials/01-foundations/03-your-first-run.md) — Run and understand output

### Domain Knowledge
- [Trading for RL Practitioners](docs/tutorials/02-domains/track-a-trading-for-rl/01-trading-basics.md)
- [RL for Traders](docs/tutorials/02-domains/track-b-rl-for-traders/01-rl-fundamentals.md)
- [Common Failures](docs/tutorials/02-domains/track-b-rl-for-traders/02-common-failures.md) — Critical pitfalls to avoid
- [Full Introduction](docs/tutorials/02-domains/track-c-full-intro/README.md) — New to both domains

### Core Components
- [Action Schemes](docs/tutorials/03-components/01-action-schemes.md) — BSH and order execution
- [Reward Schemes](docs/tutorials/03-components/02-reward-schemes.md) — Why PBR works
- [Observers & Feeds](docs/tutorials/03-components/03-observers-feeds.md) — Feature engineering

### Training
- [First Training](docs/tutorials/04-training/01-first-training.md) — Train with Ray RLlib
- [Ray RLlib Deep Dive](docs/tutorials/04-training/02-ray-rllib.md) — Configuration options
- [Optuna Optimization](docs/tutorials/04-training/03-optuna.md) — Hyperparameter tuning

### Advanced Topics
- [Overfitting](docs/tutorials/05-advanced/01-overfitting.md) — Detection and prevention
- [Commission Analysis](docs/tutorials/05-advanced/02-commission.md) — Key research findings
- [Walk-Forward Validation](docs/tutorials/05-advanced/03-walk-forward.md) — Proper evaluation

### Additional Resources
- [Experiments Log](docs/EXPERIMENTS.md) — Full research documentation
- [Environment Setup](docs/ENVIRONMENT_SETUP.md) — Detailed installation guide
- [API Reference](https://www.tensortrade.org/en/latest/)

---

## Research Findings

We conducted extensive experiments training PPO agents on BTC/USD. Key results:

| Configuration | Test P&L | vs Buy-and-Hold |
|---------------|----------|-----------------|
| Agent (0% commission) | +$239 | +$594 |
| Agent (0.1% commission) | -$650 | -$295 |
| Buy-and-Hold | -$355 | — |

The agent demonstrates directional prediction capability at zero commission. The primary challenge is trading frequency—commission costs currently exceed prediction profits. See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for methodology and detailed analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TradingEnv                               │
│                                                                 │
│   Observer ──────> Agent ──────> ActionScheme ──────> Portfolio │
│   (features)      (policy)      (BSH/Orders)        (wallets)  │
│       ^                                                  │      │
│       └──────────── RewardScheme <───────────────────────┘      │
│                        (PBR)                                    │
│                                                                 │
│   DataFeed ──────> Exchange ──────> Broker ──────> Trades       │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Default |
|-----------|---------|---------|
| ActionScheme | Converts agent output to orders | BSH (Buy/Sell/Hold) |
| RewardScheme | Computes learning signal | PBR (Position-Based Returns) |
| Observer | Generates observations | Windowed features |
| Portfolio | Manages wallets and positions | USD + BTC |
| Exchange | Simulates execution | Configurable commission |

---

## Training Scripts

| Script | Description |
|--------|-------------|
| `examples/training/train_simple.py` | Basic demo with wallet tracking |
| `examples/training/train_ray_long.py` | Distributed training with Ray RLlib |
| `examples/training/train_optuna.py` | Hyperparameter optimization |
| `examples/training/train_best.py` | Best configuration from experiments |

---

## Installation

**Requirements:** Python 3.11 or 3.12

```bash
# Create environment
python3.12 -m venv tensortrade-env
source tensortrade-env/bin/activate  # Windows: tensortrade-env\Scripts\activate

# Install
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Verify
pytest tests/tensortrade/unit -v

# Training dependencies (optional)
pip install -r examples/requirements.txt
```

See [ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for platform-specific instructions and troubleshooting.

### Docker

```bash
make run-notebook  # Jupyter
make run-docs      # Documentation
make run-tests     # Test suite
```

---

## Project Structure

```
tensortrade/
├── tensortrade/           # Core library
│   ├── env/              # Trading environments
│   ├── feed/             # Data pipeline
│   ├── oms/              # Order management
│   └── data/             # Data fetching
├── examples/
│   ├── training/         # Training scripts
│   └── notebooks/        # Jupyter tutorials
├── docs/
│   ├── tutorials/        # Learning curriculum
│   └── EXPERIMENTS.md    # Research log
└── tests/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No stream satisfies selector" | Update to v1.0.4-dev1+ |
| Ray installation fails | Run `pip install --upgrade pip` first |
| NumPy version conflict | `pip install "numpy>=1.26.4,<2.0"` |
| TensorFlow CUDA issues | `pip install tensorflow[and-cuda]>=2.15.1` |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
1. Trading frequency reduction (position sizing, holding periods)
2. Commission-aware reward schemes
3. Alternative action spaces

---

## Community

- [Discord](https://discord.gg/ZZ7BGWh)
- [GitHub Issues](https://github.com/notadamking/tensortrade/issues)
- [Documentation](https://www.tensortrade.org/)

---

## License

[Apache 2.0](LICENSE)
