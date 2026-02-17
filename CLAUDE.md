# TensorTrade - Claude Code Instructions

## Python Environment (uv)

```bash
# Create venv (one-time)
uv venv --python 3.12 .venv

# Install all deps (uv workspace)
uv sync --all-extras --group dev

# Activate (if needed outside uv)
source .venv/bin/activate
```

Always use `.venv/bin/python` to run Python commands, or prefix with `uv run`.

## Frontend Setup

```bash
cd dashboard && npm install
```
Run this after cloning or switching branches. `dev.sh` auto-installs if `node_modules/` is missing.

## Running Servers

### Quick start (both servers)
```bash
make dev        # or: ./dev.sh
```
Starts backend (FastAPI :8000) + frontend (Next.js :3000) in background. Loads `.env` automatically. PIDs tracked in `.dev-pids/`, logs in `.dev-logs/`.

```bash
make stop       # or: ./dev.sh stop
make dev-status # or: ./dev.sh status
```

### Individual servers (if needed)
```bash
# Backend only
.venv/bin/python -m tensortrade_platform.api.server

# Frontend only
cd dashboard && npm run dev
```
The backend app uses a factory function `create_app()` — there is no module-level `app` variable. Use `python -m` to run it, not `uvicorn tensortrade_platform.api.server:app`. The frontend proxies API/WebSocket calls to port 8000.

### Environment variables
Put secrets in `.env` at project root (git-ignored). The backend loads it on startup.
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Running Tests

### Python tests
```bash
uv run pytest packages/tensortrade/tests/ -x -q          # core only
uv run pytest packages/tensortrade-platform/tests/ -x -q  # platform only
make test                                                  # all tests
```

### Frontend tests
```bash
cd dashboard && npm run test
```

### Python lint/format
```bash
make lint       # ruff check packages/
make format     # ruff format + auto-fix
```

### Frontend lint/format
```bash
cd dashboard && npx biome check --write src/
```

## Project Structure (uv workspace)

```
packages/
  tensortrade/          — core RL library (import as tensortrade.*)
    tensortrade/        — env, agents, feed, oms, core, stochastic, contrib
    tests/              — core tests
  tensortrade-platform/ — platform infrastructure (import as tensortrade_platform.*)
    tensortrade_platform/ — api, training, live, data, ray_config, ray_manager
    tests/              — platform tests
dashboard/              — Next.js 15 / React 19 / Zustand / Recharts frontend
examples/               — training scripts & Jupyter notebooks
```

Key files:
- `packages/tensortrade-platform/tensortrade_platform/api/server.py` — FastAPI backend (REST + WebSocket)
- `packages/tensortrade-platform/tensortrade_platform/training/` — training modules (stores, launcher, features, callbacks)
- `dashboard/src/app/` — pages (file-based routing)
- `dashboard/src/stores/` — Zustand state stores
- `dashboard/src/lib/api.ts` — typed API client
- `dashboard/src/lib/types.ts` — shared TypeScript interfaces

## Database

All training data is stored in SQLite at `~/.tensortrade/experiments.db` (WAL mode for concurrent access).

The database is **auto-initialized on server startup**: tables are created and seed data (3 datasets, 4 HP packs) is inserted if the tables are empty. No manual setup needed.

```bash
make init-db    # Check database status and seed if empty
make reset-db   # Delete database and re-seed from scratch
```

Stores that share the database:
- `ExperimentStore` — experiments, iterations, trades, optuna_trials, insights
- `DatasetStore` — dataset_configs (auto-seeds 3 defaults: Synthetic GBM, BTC Hourly, BTC Trend)
- `HyperparameterStore` — hyperparameter_packs (auto-seeds 4 defaults: Simple PPO, Best Known, Trend Following, Optuna Optimized)

## Training System

- `tensortrade_platform/training/launcher.py` — generates and spawns training scripts as subprocesses
- `tensortrade_platform/training/experiment_store.py` — SQLite store at `~/.tensortrade/experiments.db`
- `tensortrade_platform/training/dataset_store.py` — dataset configs (synthetic + crypto)
- `tensortrade_platform/training/hyperparameter_store.py` — HP pack presets
- `tensortrade_platform/training/feature_engine.py` — computes features (returns, RSI, SMA, volatility, etc.)

### WebSocket Architecture
- Training subprocess connects to `/ws/training` (producer)
- Dashboard connects to `/ws/dashboard` (consumer)
- Server forwards messages from producer to all dashboard clients
- Message types: `experiment_start`, `training_update`, `training_progress`, `episode_metrics`, `experiment_end`

### RLlib Notes (Ray 2.53)
- Uses old API stack: `api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)`
- `EpisodeV2` does NOT have `last_action_for()` — track actions via env wrapper instead
- Metrics at `result["env_runners"]` (new) or top-level (old); check both
- RLlib returns NaN for metrics when no episodes complete; always use `safe_float()` helper

## Key Technical Details

- Python 3.12+, TensorFlow 2.20, Ray 2.53, numpy 1.26.4, pandas 2.3.3, gymnasium 1.1.1
- `env.action_space.n` returns numpy int; Keras needs `int()` cast
- Gymnasium returns 5-tuple from `step()` (not 4)
- Use `ta` library (not `pandas_ta` — incompatible with numpy <2.2.6)
- Never use `any` type in TypeScript — create proper interfaces
- Run `make lint` after editing Python code (ruff)
- Run `npx biome check --write src/` after editing dashboard code
- Docker changes need `docker rmi tensortrade:latest` before rebuild
