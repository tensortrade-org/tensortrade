# TensorTrade - Claude Code Instructions

## Python Environment (uv)

```bash
# Create venv (one-time)
uv venv --python 3.12 .venv

# Install all deps
uv pip install -e ".[dashboard,insights,optuna]"

# Activate (if needed outside uv)
source .venv/bin/activate
```

Always use `.venv/bin/python` to run Python commands, or prefix with `uv run`.

## Running Servers

### Backend API (FastAPI on port 8000)
```bash
.venv/bin/python -m tensortrade.api.server
```
The app uses a factory function `create_app()` — there is no module-level `app` variable. Use `python -m` to run it, not `uvicorn tensortrade.api.server:app`.

### Frontend Dashboard (Next.js on port 3000)
```bash
cd dashboard && npm run dev
```

### Both together
Start backend first, then frontend. The frontend proxies API/WebSocket calls to port 8000.

## Running Tests

### Python tests
```bash
.venv/bin/python -m pytest tests/ -x -q
```

### Frontend tests
```bash
cd dashboard && npm run test
```

### Frontend lint/format
```bash
cd dashboard && npx biome check --write src/
```

## Project Structure

- `tensortrade/` — core library (env, agents, feeds, oms)
- `tensortrade/api/server.py` — FastAPI backend (REST + WebSocket)
- `tensortrade/training/` — training modules (stores, launcher, features, callbacks)
- `dashboard/` — Next.js 15 / React 19 / Zustand / Recharts frontend
- `dashboard/src/app/` — pages (file-based routing)
- `dashboard/src/stores/` — Zustand state stores
- `dashboard/src/lib/api.ts` — typed API client
- `dashboard/src/lib/types.ts` — shared TypeScript interfaces
- `examples/training/` — training scripts
- `tests/` — pytest test suite

## Key Technical Details

- Python 3.12+, TensorFlow 2.20, numpy 1.26.4, pandas 2.3.3, gymnasium 1.1.1
- `env.action_space.n` returns numpy int; Keras needs `int()` cast
- Gymnasium returns 5-tuple from `step()` (not 4)
- Use `ta` library (not `pandas_ta` — incompatible with numpy <2.2.6)
- Never use `any` type in TypeScript — create proper interfaces
- Run `npx biome check --write src/` after editing dashboard code
