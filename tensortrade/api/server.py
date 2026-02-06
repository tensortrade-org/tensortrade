"""
FastAPI server for the TensorTrade training dashboard.

Provides REST endpoints for experiment data and WebSocket
endpoints for real-time training data streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from tensortrade.training.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for dashboard clients and training producers."""

    def __init__(self) -> None:
        self.dashboard_clients: list[WebSocket] = []
        self.training_producers: list[WebSocket] = []
        self._is_training = False
        self._is_paused = False
        self._current_experiment_id: str | None = None
        self._current_iteration = 0

    async def connect_dashboard(self, ws: WebSocket) -> None:
        await ws.accept()
        self.dashboard_clients.append(ws)

    async def connect_training(self, ws: WebSocket) -> None:
        await ws.accept()
        self.training_producers.append(ws)
        self._is_training = True

    def disconnect_dashboard(self, ws: WebSocket) -> None:
        if ws in self.dashboard_clients:
            self.dashboard_clients.remove(ws)

    def disconnect_training(self, ws: WebSocket) -> None:
        if ws in self.training_producers:
            self.training_producers.remove(ws)
        if not self.training_producers:
            self._is_training = False

    async def broadcast_to_dashboards(self, message: dict) -> None:
        """Send a message to all connected dashboard clients."""
        disconnected: list[WebSocket] = []
        for client in self.dashboard_clients:
            try:
                await client.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                disconnected.append(client)
        for client in disconnected:
            self.disconnect_dashboard(client)

    async def send_control_to_training(self, command: str) -> None:
        """Send a control command to all training producers."""
        for producer in self.training_producers:
            try:
                await producer.send_json({"command": command})
            except (WebSocketDisconnect, RuntimeError):
                pass

    def get_status(self) -> dict:
        return {
            "type": "status",
            "is_training": self._is_training,
            "is_paused": self._is_paused,
            "experiment_id": self._current_experiment_id,
            "current_iteration": self._current_iteration,
            "dashboard_clients": len(self.dashboard_clients),
            "training_producers": len(self.training_producers),
        }


# Module-level state
_manager = ConnectionManager()
_store: ExperimentStore | None = None


def _get_store() -> ExperimentStore:
    global _store
    if _store is None:
        _store = ExperimentStore()
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _store
    _store = ExperimentStore()
    logger.info("TensorTrade API server started")
    yield
    if _store:
        _store.close()
    logger.info("TensorTrade API server stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TensorTrade Dashboard API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Register all REST and WebSocket routes."""

    # --- REST: Experiments ---

    @app.get("/api/experiments")
    async def list_experiments(
        script: str | None = None,
        status: str | None = None,
        limit: int = Query(default=100, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        store = _get_store()
        experiments = store.list_experiments(
            script=script, status=status, limit=limit, offset=offset
        )
        return [asdict(e) for e in experiments]

    @app.get("/api/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str) -> dict:
        store = _get_store()
        exp = store.get_experiment(experiment_id)
        if not exp:
            return {"error": "not found"}
        iterations = store.get_iterations(experiment_id)
        return {
            "experiment": asdict(exp),
            "iterations": [asdict(it) for it in iterations],
        }

    @app.get("/api/experiments/{experiment_id}/trades")
    async def get_experiment_trades(
        experiment_id: str,
        episode: int | None = None,
        limit: int = Query(default=1000, le=5000),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        store = _get_store()
        trades = store.get_trades(
            experiment_id, episode=episode, limit=limit, offset=offset
        )
        return [asdict(t) for t in trades]

    # --- REST: Leaderboard ---

    @app.get("/api/leaderboard")
    async def get_leaderboard(
        metric: str = Query(default="pnl"),
        script: str | None = None,
        limit: int = Query(default=50, le=200),
    ) -> list[dict]:
        store = _get_store()
        entries = store.get_leaderboard(metric=metric, script=script, limit=limit)
        return [asdict(e) for e in entries]

    # --- REST: Optuna ---

    @app.get("/api/optuna/studies")
    async def list_optuna_studies() -> list[dict]:
        store = _get_store()
        return store.get_optuna_studies()

    @app.get("/api/optuna/studies/{name}")
    async def get_optuna_study(name: str) -> dict:
        store = _get_store()
        trials = store.get_optuna_trials(name)
        return {
            "study_name": name,
            "trials": [asdict(t) for t in trials],
            "total": len(trials),
            "completed": sum(1 for t in trials if t.state == "complete"),
            "pruned": sum(1 for t in trials if t.state == "pruned"),
        }

    @app.get("/api/optuna/studies/{name}/importance")
    async def get_param_importance(name: str) -> dict:
        """Get parameter importance (requires optuna study object).

        Falls back to trial-based heuristic if study not available.
        """
        store = _get_store()
        trials = store.get_optuna_trials(name)
        if not trials:
            return {"error": "No trials found"}

        # Compute simple variance-based importance from stored trials
        complete_trials = [t for t in trials if t.state == "complete" and t.value is not None]
        if len(complete_trials) < 3:
            return {"importance": {}, "note": "Need at least 3 complete trials"}

        param_keys = set()
        for t in complete_trials:
            param_keys.update(t.params.keys())

        importance: dict[str, float] = {}
        for key in param_keys:
            values = []
            objectives = []
            for t in complete_trials:
                if key in t.params:
                    val = t.params[key]
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                        objectives.append(float(t.value))  # type: ignore[arg-type]

            if len(values) >= 3:
                # Correlation-based importance
                corr = abs(_correlation(values, objectives))
                importance[key] = round(corr, 4)

        # Normalize
        total = sum(importance.values()) or 1.0
        importance = {k: round(v / total, 4) for k, v in importance.items()}
        return {"importance": dict(sorted(importance.items(), key=lambda x: -x[1]))}

    @app.get("/api/optuna/studies/{name}/curves")
    async def get_study_curves(name: str) -> dict:
        """Get all trials for a study with per-iteration training curves."""
        store = _get_store()
        trials = store.get_study_trial_curves(name)
        return {"study_name": name, "trials": trials}

    # --- REST: Inference ---

    @app.post("/api/inference/run")
    async def run_inference(body: dict) -> dict:
        """Start an inference episode in the background."""
        from tensortrade.api.inference_runner import InferenceRunner

        experiment_id = body.get("experiment_id")
        use_random_agent = body.get("use_random_agent", True)
        if not experiment_id:
            return {"error": "experiment_id is required"}

        store = _get_store()
        runner = InferenceRunner(store, _manager)
        asyncio.create_task(runner.run_episode(experiment_id, use_random_agent))
        return {"status": "started"}

    # --- REST: Insights ---

    @app.post("/api/insights/analyze")
    async def analyze(body: dict) -> dict:
        store = _get_store()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not set"}

        from tensortrade.api.insights import InsightsEngine

        engine = InsightsEngine(store, api_key)
        analysis_type = body.get("type", "experiment")

        try:
            if analysis_type == "experiment":
                report = await engine.analyze_experiment(body["experiment_id"])
            elif analysis_type == "comparison":
                report = await engine.compare_experiments(body["experiment_ids"])
            elif analysis_type == "strategy":
                report = await engine.suggest_next_strategy(body["study_name"])
            elif analysis_type == "trades":
                report = await engine.analyze_trades(body["experiment_id"])
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

            return asdict(report)
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/insights/{insight_id}")
    async def get_insight(insight_id: str) -> dict:
        store = _get_store()
        insight = store.get_insight(insight_id)
        if not insight:
            return {"error": "not found"}
        return insight

    @app.get("/api/insights")
    async def list_insights(limit: int = Query(default=50, le=200)) -> list[dict]:
        store = _get_store()
        return store.list_insights(limit=limit)

    # --- REST: Trades ---

    @app.get("/api/trades")
    async def list_all_trades(
        experiment_id: str | None = None,
        side: str | None = None,
        limit: int = Query(default=1000, le=5000),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        store = _get_store()
        return store.get_all_trades(
            limit=limit, offset=offset, experiment_id=experiment_id, side=side
        )

    # --- REST: Status ---

    @app.get("/api/status")
    async def get_status() -> dict:
        return _manager.get_status()

    # --- REST: Training Controls ---

    @app.post("/api/training/stop")
    async def stop_training() -> dict:
        await _manager.send_control_to_training("stop")
        return {"status": "stop_sent"}

    @app.post("/api/training/pause")
    async def pause_training() -> dict:
        _manager._is_paused = True
        await _manager.send_control_to_training("pause")
        return {"status": "pause_sent"}

    @app.post("/api/training/resume")
    async def resume_training() -> dict:
        _manager._is_paused = False
        await _manager.send_control_to_training("resume")
        return {"status": "resume_sent"}

    # --- WebSocket: Dashboard (consumer) ---

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(ws: WebSocket) -> None:
        await _manager.connect_dashboard(ws)
        try:
            # Send current status on connect
            await ws.send_json(_manager.get_status())
            while True:
                # Keep connection alive, listen for commands
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                    cmd = msg.get("command")
                    if cmd in ("stop", "pause", "resume"):
                        await _manager.send_control_to_training(cmd)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            _manager.disconnect_dashboard(ws)

    # --- WebSocket: Training (producer) ---

    @app.websocket("/ws/training")
    async def ws_training(ws: WebSocket) -> None:
        await _manager.connect_training(ws)
        try:
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                    # Track iteration progress
                    if msg.get("type") == "training_update":
                        _manager._current_iteration = msg.get("iteration", 0)
                    if msg.get("type") == "experiment_start":
                        _manager._current_experiment_id = msg.get("experiment_id")

                    # Forward to all dashboard clients
                    await _manager.broadcast_to_dashboards(msg)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            _manager.disconnect_training(ws)
            await _manager.broadcast_to_dashboards({
                "type": "training_disconnected",
            })


def _correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / (n - 1)) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / (n - 1)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sx * sy)


# Convenience: run with `python -m tensortrade.api.server`
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
