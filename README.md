<p align="center">
  <img src="docs/source/_static/logo.jpg" width="200" alt="TensorTrade Logo">
</p>

# TensorTrade

**An open-source Python framework for training reinforcement learning agents to trade.**

[![Tests](https://github.com/tensortrade-org/tensortrade/actions/workflows/tests.yml/badge.svg)](https://github.com/tensortrade-org/tensortrade/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/tensortrade/badge/?version=latest)](https://tensortrade.org)
[![Apache License](https://img.shields.io/github/license/tensortrade-org/tensortrade.svg?color=brightgreen)](http://www.apache.org/licenses/LICENSE-2.0)
[![Discord](https://img.shields.io/discord/592446624882491402.svg?color=brightgreen)](https://discord.gg/ZZ7BGWh)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

TensorTrade provides composable building blocks for trading environments, action schemes, reward functions, and data feeds. Wire them together, point an RL algorithm at the environment, and see if your agent can beat buy-and-hold.

```python
import tensortrade.env.default as default
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

# Price data (swap in real OHLCV data here)
price = Stream.source([100 + i * 0.5 for i in range(200)], dtype="float").rename("USD-BTC")

exchange = Exchange("sim", service=execute_order)(price)
portfolio = Portfolio(USD, [
    Wallet(exchange, 10_000 * USD),
    Wallet(exchange, 0 * BTC),
])

env = default.create(
    portfolio=portfolio,
    action_scheme="bsh",       # Buy / Sell / Hold
    reward_scheme="pbr",       # Position-Based Returns
    window_size=20,
)

# env is a standard gymnasium.Env - use with any RL library
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Install

```bash
pip install tensortrade          # core library only
pip install tensortrade-platform # + training dashboard, live trading, API server
```

**From source (development):**

```bash
git clone https://github.com/tensortrade-org/tensortrade.git && cd tensortrade
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync --all-extras --group dev    # installs both packages + test/lint deps
uv run pytest packages/tensortrade/tests/ -x -q   # verify
```

## Architecture

```
                          TradingEnv (gymnasium.Env)
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Observer ────> Agent ────> ActionScheme ────> Portfolio     │
  │   (features)    (policy)    (BSH / Orders)    (wallets)      │
  │       ^                                           |          │
  │       └──────── RewardScheme <────────────────────┘          │
  │                    (PBR)                                     │
  │                                                              │
  │   DataFeed ────> Exchange ────> Broker ────> Trades          │
  └──────────────────────────────────────────────────────────────┘
```

| Component | What it does | Default |
|-----------|-------------|---------|
| **ActionScheme** | Maps agent output to orders | BSH (Buy/Sell/Hold) |
| **RewardScheme** | Computes learning signal | PBR (Position-Based Returns) |
| **Observer** | Builds observation window | Windowed price features |
| **Portfolio** | Tracks wallets and positions | USD + BTC |
| **Exchange** | Simulates order execution | Configurable commission |

## Packages

This repo is a **uv workspace** with two Python packages:

| Package | What | Install |
|---------|------|---------|
| [`tensortrade`](packages/tensortrade/) | Core RL trading library (env, feed, oms, agents, stochastic) | `pip install tensortrade` |
| [`tensortrade-platform`](packages/tensortrade-platform/) | Training infra, API server, live trading, data fetching | `pip install tensortrade-platform` |

The core library has **zero** dependency on the platform. You can use `tensortrade` on its own with any RL library (Stable-Baselines3, CleanRL, RLlib, etc).

```
packages/
  tensortrade/                # Core library
    tensortrade/
      core/                   # Base classes, clock, context, registry
      feed/                   # Streaming data pipeline
      oms/                    # Order management (exchanges, wallets, orders)
      env/                    # Gymnasium environments + default components
      agents/                 # Built-in DQN / A2C agents
      stochastic/             # Synthetic price generators (GBM, Heston, etc.)
  tensortrade-platform/       # Platform (optional)
    tensortrade_platform/
      api/                    # FastAPI + WebSocket server
      training/               # Ray/RLlib launcher, stores, callbacks
      live/                   # Alpaca live/paper trading
      data/                   # Crypto data fetching
```

## Documentation

**[Tutorial Index](docs/tutorials/index.md)** — the full learning curriculum.

| Section | Topics |
|---------|--------|
| [Foundations](docs/tutorials/01-foundations/) | RL + trading concepts, architecture, your first run |
| [Domain Knowledge](docs/tutorials/02-domains/) | Trading for RL practitioners, RL for traders, common failures |
| [Components](docs/tutorials/03-components/) | Action schemes, reward schemes, observers and feeds |
| [Training](docs/tutorials/04-training/) | Ray RLlib, Optuna hyperparameter tuning |
| [Advanced](docs/tutorials/05-advanced/) | Overfitting, commission analysis, walk-forward validation |
| [Experiments](docs/EXPERIMENTS.md) | Full research log with results |
| [API Reference](https://www.tensortrade.org/en/latest/) | Auto-generated from source |

## Training Dashboard (Platform)

The platform package includes a real-time dashboard for launching and monitoring training runs.

```bash
make dev          # starts backend (:8000) + frontend (:3000)
make stop         # stop both
```

Features: live training progress, iteration metrics charts, action distribution, experiment history, dataset and hyperparameter selection.

See the [platform package README](packages/tensortrade-platform/) for setup details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Community

- [Discord](https://discord.gg/ZZ7BGWh)
- [GitHub Issues](https://github.com/tensortrade-org/tensortrade/issues)
- [Documentation](https://www.tensortrade.org/)

## License

[Apache 2.0](LICENSE)
