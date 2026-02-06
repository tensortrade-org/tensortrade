"""Tests for the FastAPI dashboard server."""

import pytest
from unittest.mock import patch

from tensortrade.api.server import create_app, _manager, ConnectionManager


@pytest.fixture
def store(tmp_path):
    from tensortrade.training.experiment_store import ExperimentStore
    db_path = str(tmp_path / "test_server.db")
    s = ExperimentStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def app(store):
    """Create a FastAPI app with a test store."""
    import tensortrade.api.server as server_module

    original_store = server_module._store
    server_module._store = store
    application = create_app()
    yield application
    server_module._store = original_store


@pytest.fixture
def client(app):
    """Create a test client."""
    from starlette.testclient import TestClient

    return TestClient(app, raise_server_exceptions=False)


class TestExperimentEndpoints:
    def test_list_experiments_empty(self, client):
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_experiments(self, client, store):
        store.create_experiment(name="run1", script="test")
        store.create_experiment(name="run2", script="test")

        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_experiments_filter_script(self, client, store):
        store.create_experiment(name="run1", script="train_a")
        store.create_experiment(name="run2", script="train_b")

        resp = client.get("/api/experiments?script=train_a")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["script"] == "train_a"

    def test_get_experiment(self, client, store):
        eid = store.create_experiment(name="run1", script="test", config={"lr": 0.001})
        store.log_iteration(eid, 1, {"pnl": 50.0})

        resp = client.get(f"/api/experiments/{eid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment"]["name"] == "run1"
        assert len(data["iterations"]) == 1

    def test_get_experiment_not_found(self, client):
        resp = client.get("/api/experiments/nonexistent")
        data = resp.json()
        assert "error" in data

    def test_get_experiment_trades(self, client, store):
        eid = store.create_experiment(name="run1", script="test")
        store.log_trade(eid, episode=0, step=10, side="buy", price=100.0, size=0.5)

        resp = client.get(f"/api/experiments/{eid}/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["side"] == "buy"


class TestLeaderboardEndpoint:
    def test_leaderboard(self, client, store):
        for i, pnl in enumerate([100, 500, 250]):
            eid = store.create_experiment(name=f"run{i}", script="test")
            store.complete_experiment(eid, final_metrics={"pnl": pnl})

        resp = client.get("/api/leaderboard?metric=pnl")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["metric_value"] == 500

    def test_leaderboard_filter_script(self, client, store):
        for script, pnl in [("a", 100), ("b", 200)]:
            eid = store.create_experiment(name=f"run_{script}", script=script)
            store.complete_experiment(eid, final_metrics={"pnl": pnl})

        resp = client.get("/api/leaderboard?metric=pnl&script=a")
        data = resp.json()
        assert len(data) == 1


class TestOptunaEndpoints:
    def test_list_studies(self, client, store):
        store.log_optuna_trial("study_a", 0, {"lr": 0.001}, 0.8, "complete", 60.0)
        store.log_optuna_trial("study_b", 0, {"lr": 0.005}, 0.9, "complete", 90.0)

        resp = client.get("/api/optuna/studies")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_get_study_detail(self, client, store):
        store.log_optuna_trial("my_study", 0, {"lr": 0.001}, 0.8, "complete", 60.0)
        store.log_optuna_trial("my_study", 1, {"lr": 0.01}, 0.7, "pruned", 30.0)

        resp = client.get("/api/optuna/studies/my_study")
        assert resp.status_code == 200
        data = resp.json()
        assert data["study_name"] == "my_study"
        assert data["total"] == 2
        assert data["completed"] == 1
        assert data["pruned"] == 1

    def test_param_importance_insufficient_trials(self, client, store):
        store.log_optuna_trial("small_study", 0, {"lr": 0.001}, 0.8, "complete", 60.0)

        resp = client.get("/api/optuna/studies/small_study/importance")
        data = resp.json()
        assert "note" in data or "importance" in data

    def test_param_importance(self, client, store):
        # Need at least 3 complete trials with numeric params
        for i, (lr, val) in enumerate([(0.001, 0.5), (0.01, 0.7), (0.1, 0.9), (0.05, 0.8)]):
            store.log_optuna_trial("corr_study", i, {"lr": lr}, val, "complete", 60.0)

        resp = client.get("/api/optuna/studies/corr_study/importance")
        data = resp.json()
        assert "importance" in data
        assert "lr" in data["importance"]


class TestTradesEndpoint:
    def test_list_all_trades(self, client, store):
        eid = store.create_experiment(name="run1", script="test")
        store.log_trade(eid, 0, 10, "buy", 100.0, 0.5, 0.1)
        store.log_trade(eid, 0, 20, "sell", 110.0, 0.5, 0.1)

        resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_filter_trades_by_side(self, client, store):
        eid = store.create_experiment(name="run1", script="test")
        store.log_trade(eid, 0, 10, "buy", 100.0, 0.5)
        store.log_trade(eid, 0, 20, "sell", 110.0, 0.5)

        resp = client.get("/api/trades?side=buy")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["side"] == "buy"


class TestStatusEndpoint:
    def test_get_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_training" in data
        assert "is_paused" in data
        assert "dashboard_clients" in data


class TestTrainingControls:
    def test_stop_training(self, client):
        resp = client.post("/api/training/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stop_sent"

    def test_pause_training(self, client):
        resp = client.post("/api/training/pause")
        assert resp.status_code == 200

    def test_resume_training(self, client):
        resp = client.post("/api/training/resume")
        assert resp.status_code == 200


class TestInsightsEndpoints:
    def test_list_insights_empty(self, client):
        resp = client.get("/api/insights")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_insight(self, client, store):
        eid = store.create_experiment(name="run1", script="test")
        store.store_insight(
            insight_id="test-insight",
            experiment_ids=[eid],
            analysis_type="experiment",
            summary="Test summary",
            findings=["finding1"],
            suggestions=["suggestion1"],
            confidence="high",
            raw_response="raw",
        )

        resp = client.get("/api/insights/test-insight")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == "Test summary"

    def test_get_insight_not_found(self, client):
        resp = client.get("/api/insights/nonexistent")
        data = resp.json()
        assert "error" in data

    def test_analyze_no_api_key(self, client):
        with patch.dict("os.environ", {}, clear=True):
            resp = client.post(
                "/api/insights/analyze",
                json={"type": "experiment", "experiment_id": "test"},
            )
            data = resp.json()
            assert "error" in data


class TestConnectionManager:
    def test_initial_status(self):
        mgr = ConnectionManager()
        status = mgr.get_status()
        assert status["is_training"] is False
        assert status["is_paused"] is False
        assert status["dashboard_clients"] == 0
        assert status["training_producers"] == 0


class TestCorrelationHelper:
    def test_correlation(self):
        from tensortrade.api.server import _correlation

        # Perfect positive correlation
        assert abs(_correlation([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-10

        # Perfect negative correlation
        assert abs(_correlation([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-10

    def test_correlation_insufficient_data(self):
        from tensortrade.api.server import _correlation

        assert _correlation([1], [2]) == 0.0

    def test_correlation_constant_values(self):
        from tensortrade.api.server import _correlation

        assert _correlation([5, 5, 5], [1, 2, 3]) == 0.0
