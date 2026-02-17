# Pre-Merge TODO: feature/training-dashboard

330 files changed, ~47k lines added. Core library and training pipeline are solid (426+ tests passing), but the new surface area — API server, live trading, dashboard, WebSocket layer, and insights engine — needs auth, tests, and hardening before production.

---

## P0 — Blockers

- [ ] **Authentication & authorization** — All API routes are wide open with `allow_origins=["*"]`. Need at minimum API key auth or JWT before anyone can hit `/api/live/start` and trade real money. (`server.py:186-203`)
- [ ] **Live trading has zero tests** — `trader.py` (683 LOC), `store.py`, `account_sync.py`, `config.py` — none have test files. This is the code that submits real Alpaca orders.
- [ ] **WebSocket URL uses hardcoded `ws://`** — `DashboardShell.tsx:161` doesn't detect HTTPS. Will break behind any TLS terminator. Needs `wss:` when on HTTPS.
- [ ] **File handle leak in launcher** — `launcher.py:117` opens log files with `open()` passed to `Popen` but never closes them. Will exhaust file descriptors on long-running servers.
- [ ] **Position tracking race condition** — `trader.py` updates `self._position = 1` before the DB commit. A crash between the two means position state diverges from the database.
- [ ] **No WebSocket endpoint tests** — `/ws/dashboard`, `/ws/training`, `/ws/live` have 0% coverage. These are the primary data path for the UI.

## P1 — High Priority

- [ ] **SQLite thread safety** — All stores use `check_same_thread=False` with no mutex. Concurrent FastAPI requests + training subprocesses writing to the same DB will cause locking errors under load.
- [ ] **API returns 200 for not-found** — Multiple endpoints return `{"error": "not found"}` with HTTP 200 instead of raising `HTTPException(404)`. Breaks REST conventions and client error handling.
- [ ] **Insights engine untested** — `api/insights.py` (712 LOC) handles Anthropic API streaming, prompt construction, and HP pack generation with zero test coverage.
- [ ] **Training bridge untested** — `api/training_bridge.py` (168 LOC) does threading + asyncio + WebSocket reconnection with no tests.
- [ ] **Missing input validation on script generation** — `launcher.py:330` embeds user config directly into generated Python scripts via f-string. Needs proper escaping or safer serialization.
- [ ] **Order idempotency gap** — `alpaca.py` generates a UUID `client_order_id` but doesn't persist it. Network retry could create duplicate orders.
- [ ] **Dashboard test coverage at 4%** — 4 store tests out of 93 components/pages. At minimum need tests for: live trading page, launch wizard, campaign orchestration, insights request flow.
- [ ] **No error boundaries** — No React error boundary in `layout.tsx`. Unhandled component errors show a white screen.
- [ ] **Unbounded array growth in inference store** — `inferenceStore.ts` `addStep()` has no `MAX_STEPS` limit unlike the training store. Will leak memory in long sessions.

## P2 — Should Fix

- [ ] **WebSocket reconnection has no backoff** — Fixed 3-second interval in `useWebSocket.ts`. Needs exponential backoff to avoid hammering the server on mass disconnects.
- [ ] **No fetch timeouts** — All `fetch()` calls in `api.ts` have no timeout configuration. Can hang indefinitely if backend is slow.
- [ ] **Type safety violations** — Multiple `as unknown as Record<string, unknown>` casts in `api.ts` (lines 204, 226, 239, 270, etc.). Need proper request/response interfaces.
- [ ] **Generic exception handling** — `alpaca.py`, `account_sync.py`, `trader.py` all use bare `except Exception:` with only logging. Need specific exception types, backoff, and alerting.
- [ ] **Account reconciliation tolerance** — `account_sync.py:136` uses `1e-8` tolerance for position mismatch, too tight for partial fills. No auto-recovery logic.
- [ ] **Missing async task error handling** — `server.py:342` uses `asyncio.create_task()` for inference/training with no exception handler. Failures go unnoticed.
- [ ] **No rate limiting** — API has no rate limiting. A runaway client could spam `/api/training/launch` or `/api/live/start`.
- [ ] **Campaign, Optuna bridge, and callbacks untested** — `optuna_bridge.py` (153 LOC), `callbacks.py` (112 LOC) have no tests.

## P3 — Nice to Have

- [ ] **Accessibility** — Missing ARIA labels on interactive elements, incomplete keyboard navigation in DataTable.
- [ ] **Chart performance** — `MetricsLineChart` re-renders on every training update. Needs `React.memo` or downsampling.
- [ ] **No offline detection** — App doesn't detect when user goes offline; WebSocket reconnects infinitely.
- [ ] **No deployment/ops docs** — No guide for required env vars, production config, or monitoring setup.
- [ ] **Database migration strategy** — No schema versioning. Adding columns later will require manual migration.

## Continuous Training — Feature ([plan](docs/continuous-training-plan.md))

Background scheduler that periodically retrains the model on latest Alpaca candles (rolling window) and hot-swaps the live trader's policy.

### Backend
- [ ] **DB schema** — Add `continuous_training_schedules` table + CRUD methods to `experiment_store.py`
- [ ] **LiveTrader hot-swap** — Add `reload_policy()` + `_load_policy_weights_only()` to `trader.py`
- [ ] **Launcher extension** — Add `launch_continuous()` + checkpoint-restore script generation to `launcher.py`
- [ ] **ContinuousTrainer module** — New `tensortrade/training/continuous_trainer.py` with scheduler loop, cycle logic, WS broadcasting
- [ ] **API endpoints** — 6 routes in `server.py`: start/stop/pause/resume/status/history for `/api/continuous/*`

### Frontend
- [ ] **TypeScript types** — `ContinuousTrainingConfig`, `ContinuousTrainingStatus`, `ContinuousCycleRecord` + WS message types in `types.ts`
- [ ] **API client** — `startContinuousTraining()`, `stopContinuousTraining()`, etc. in `api.ts`
- [ ] **Zustand store** — `dashboard/src/stores/continuousStore.ts`
- [ ] **Dashboard page** — `dashboard/src/app/continuous/page.tsx` (setup, status, cycle history, metrics chart)
- [ ] **Sidebar nav** — Add "Continuous" link after "Paper Trading" in `Sidebar.tsx`
- [ ] **WS handler** — Handle `continuous_*` message types in dashboard WebSocket hook

### Testing
- [ ] **Python tests** — ContinuousTrainer scheduling, LiveTrader.reload_policy(), schedule CRUD
- [ ] **Lint** — `make lint` + `npx biome check --write src/`
