# Continuous Training Implementation Plan

## Context

The model currently trains once on historical data and the checkpoint is static. As markets evolve, the model becomes stale. This plan adds a background scheduler that periodically retrains the model on the latest Alpaca candles (rolling window) and hot-swaps the live trader's policy — keeping the model current without manual intervention.

**User choices**: Scheduled interval trigger, rolling data window, auto-deploy, Alpaca data source.

---

## New Files

### 1. `tensortrade/training/continuous_trainer.py` (~350 lines)

Core orchestrator. Contains:

- **`ContinuousTrainingConfig`** dataclass — `interval_hours`, `window_candles`, `iterations_per_cycle`, `fine_tune_lr_multiplier`, `symbol`, `timeframe`, `auto_deploy`
- **`ContinuousTrainer`** class:
  - `start(config)` — saves schedule to DB, launches `asyncio.Task` for `_scheduler_loop`
  - `stop()` / `pause()` / `resume()` — lifecycle control
  - `get_status()` / `get_history()` — read from DB
  - `_scheduler_loop()` — sleeps for `interval_hours`, calls `_run_cycle()`
  - `_run_cycle()`:
    1. Check `launcher.list_running()` — if busy, skip cycle and broadcast `continuous_cycle_skipped`
    2. Compute rolling window dates: `end=now`, `start=now - (window_candles * timeframe_seconds)`
    3. Create experiment with tags `["continuous", f"cycle:{N}", f"schedule:{id}"]`
    4. Call `launcher.launch_continuous()` with parent checkpoint path
    5. Poll subprocess until complete via `asyncio.sleep` loop checking `process.poll()`
    6. Read new checkpoint from experiment `final_metrics`
    7. If `auto_deploy` and `live_trader.is_running`: call `live_trader.reload_policy()`
    8. Update schedule record, broadcast `continuous_cycle_complete`
  - `_broadcast_cycle_event()` — sends WS messages via `ConnectionManager`

### 2. `dashboard/src/stores/continuousStore.ts`

Zustand store following `liveStore.ts` pattern. Tracks status, cycle history, current cycle progress.

### 3. `dashboard/src/app/continuous/page.tsx`

Dashboard page with:
- **Setup panel**: Select base experiment (with checkpoint), HP pack, dataset. Configure interval, window size, iterations, LR multiplier. Start button.
- **Status panel**: Active/paused/idle indicator, countdown to next cycle, current cycle progress bar.
- **Cycle history table**: Each cycle with experiment link, metrics, deployed indicator.
- **Metrics trend chart**: PnL/reward across cycles (Recharts line chart).
- **Controls**: Stop, Pause/Resume buttons.

---

## Modified Files

### 4. `tensortrade/training/launcher.py`

Add `launch_continuous()` method (~120 lines of new code):

```python
def launch_continuous(
    self, name, hp_pack_id, dataset_id,
    parent_checkpoint_path, cycle_number,
    continuous_config, tags=None,
) -> str:
```

- Same structure as `launch()` but calls `_generate_continuous_script()` instead
- The generated script differs from standard in 3 ways:
  1. **Restores checkpoint**: `algo.restore(PARENT_CHECKPOINT_PATH)` after `ppo_config.build()`
  2. **Reduced LR**: `lr = learning_rate * fine_tune_lr_multiplier` (default 0.2x)
  3. **Rolling window data**: Fetches via `AlpacaCryptoData().fetch(symbol, timeframe, start_date, end_date)` — no train/val/test split, entire window used for training

### 5. `tensortrade/live/trader.py`

Add policy hot-swap support:

- **`reload_policy(new_checkpoint_path) -> bool`** — loads new policy in executor, atomically swaps `self._policy`, broadcasts `live_model_updated` message. Key detail: uses a `_load_policy_weights_only()` variant that skips `ray_manager.acquire()` since the session already holds it.
- Add `_model_version: int = 1` counter to `__init__`
- Add `_broadcast_model_update()` WebSocket method

### 6. `tensortrade/training/experiment_store.py`

- Add `continuous_training_schedules` table to `_create_tables()`:
  ```sql
  CREATE TABLE IF NOT EXISTS continuous_training_schedules (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'idle',
      base_experiment_id TEXT NOT NULL,
      dataset_id TEXT NOT NULL,
      hp_pack_id TEXT NOT NULL,
      config TEXT NOT NULL DEFAULT '{}',
      current_checkpoint_path TEXT,
      current_cycle INTEGER DEFAULT 0,
      last_cycle_at TEXT,
      next_cycle_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (base_experiment_id) REFERENCES experiments(id)
  );
  ```
- Add CRUD methods: `create_continuous_schedule()`, `update_continuous_schedule()`, `get_continuous_schedule()`, `list_continuous_schedules()`

### 7. `tensortrade/api/server.py`

Add 6 REST endpoints:
- `POST /api/continuous/start` — start schedule (body: base_experiment_id, dataset_id, hp_pack_id, interval_hours, window_candles, etc.)
- `POST /api/continuous/stop` — stop schedule
- `POST /api/continuous/pause` — pause schedule
- `POST /api/continuous/resume` — resume schedule
- `GET /api/continuous/status` — current schedule status
- `GET /api/continuous/history` — cycle history with metrics

Add `_continuous_trainer` singleton, stop it in `lifespan()` shutdown.

### 8. `dashboard/src/lib/types.ts`

Add interfaces: `ContinuousTrainingConfig`, `ContinuousTrainingStatus`, `ContinuousCycleRecord`, and WS message types (`continuous_cycle_start`, `continuous_cycle_progress`, `continuous_cycle_complete`, `continuous_cycle_skipped`, `continuous_model_deployed`, `continuous_status_change`).

### 9. `dashboard/src/lib/api.ts`

Add functions: `startContinuousTraining()`, `stopContinuousTraining()`, `pauseContinuousTraining()`, `resumeContinuousTraining()`, `getContinuousStatus()`, `getContinuousHistory()`.

### 10. `dashboard/src/components/layout/Sidebar.tsx`

Add nav item after "Paper Trading":
```typescript
{ label: "Continuous", href: "/continuous", icon: "\u21BB" },
```

### 11. Dashboard WebSocket handler

Update the main WS hook to handle `continuous_*` message types and update `continuousStore`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LR for retraining | 0.2x original (configurable) | Prevents catastrophic forgetting while adapting to new data |
| Busy launcher | Skip cycle, try next interval | Avoids breaking the "one training at a time" invariant |
| Hot-swap mechanism | Atomic `self._policy` swap between bar cycles | `_process_bar` is sequential (one bar at a time via queue), no concurrent reads |
| Ray refcount | `_load_policy_weights_only()` skips `acquire()` | Session already holds "live_trading" consumer; avoids refcount leak |
| Data window | No train/val/test split | Rolling window is all "recent" data; splitting makes less sense for fine-tuning |
| Model lineage | Tags + config fields on experiment records | `parent_experiment_id` and `cycle_number` in experiment config enable chain traversal |

---

## WebSocket Message Types

```
continuous_cycle_start    - {schedule_id, cycle_number, experiment_id, window_start, window_end}
continuous_cycle_progress - {schedule_id, cycle_number, iteration, total_iterations}
continuous_cycle_complete - {schedule_id, cycle_number, experiment_id, checkpoint_path, metrics}
continuous_cycle_skipped  - {schedule_id, cycle_number, reason}
continuous_model_deployed - {schedule_id, cycle_number, checkpoint_path, model_version}
continuous_status_change  - {schedule_id, status, next_cycle_at}
```

---

## Implementation Order

1. **DB schema** — `continuous_training_schedules` table in experiment_store.py
2. **LiveTrader hot-swap** — `reload_policy()` + `_load_policy_weights_only()` in trader.py
3. **Launcher extension** — `launch_continuous()` + script generation in launcher.py
4. **ContinuousTrainer** — new module with scheduler, cycle logic, WS broadcasting
5. **API endpoints** — 6 routes in server.py
6. **TypeScript types** — interfaces in types.ts
7. **API client** — functions in api.ts
8. **Zustand store** — continuousStore.ts
9. **Dashboard page** — /continuous/page.tsx
10. **Sidebar nav** — add link
11. **WS handler** — handle continuous_* messages

---

## Verification

1. **Unit tests**: Test `ContinuousTrainer` scheduling logic (mock subprocess, mock launcher), test `LiveTrader.reload_policy()` (mock policy loading), test experiment store schedule CRUD
2. **Integration test**: Start continuous training with synthetic data, verify experiment created with correct tags, verify checkpoint saved
3. **Manual E2E test**:
   - Start a standard training run, let it complete
   - Start continuous training referencing that experiment's checkpoint
   - Verify retraining cycle fires on schedule
   - Start paper trading with the original checkpoint
   - Verify model hot-swaps after next cycle completes
   - Check dashboard shows cycle history and model version
4. **Lint**: `make lint` (Python) + `cd dashboard && npx biome check --write src/` (frontend)
5. **Existing tests pass**: `.venv/bin/python -m pytest tests/ -x -q`
