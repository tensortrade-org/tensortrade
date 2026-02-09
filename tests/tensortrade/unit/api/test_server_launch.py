"""Tests for Training Launch API endpoints."""

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from tensortrade.api.server import create_app


@pytest.fixture
def stores(tmp_path):
    db_path = str(tmp_path / "test_launch.db")

    from tensortrade.training.dataset_store import DatasetStore
    from tensortrade.training.experiment_store import ExperimentStore
    from tensortrade.training.hyperparameter_store import HyperparameterStore

    exp_store = ExperimentStore(db_path=db_path)
    hp_store = HyperparameterStore(db_path=db_path)
    ds_store = DatasetStore(db_path=db_path)
    yield exp_store, hp_store, ds_store
    exp_store.close()
    hp_store.close()
    ds_store.close()


@pytest.fixture
def mock_launcher():
    launcher = MagicMock()
    launcher.launch.return_value = "test-launched-exp-id"
    launcher.launch_campaign.return_value = "test-study"
    launcher.list_running.return_value = []
    launcher.cancel.return_value = False
    return launcher


@pytest.fixture
def client(stores, mock_launcher):
    import tensortrade.api.server as server_module

    exp_store, hp_store, ds_store = stores
    original_store = server_module._store
    original_hp = server_module._hp_store
    original_ds = server_module._ds_store
    original_launcher = server_module._launcher

    server_module._store = exp_store
    server_module._hp_store = hp_store
    server_module._ds_store = ds_store
    server_module._launcher = mock_launcher

    app = create_app()
    yield TestClient(app, raise_server_exceptions=False)

    server_module._store = original_store
    server_module._hp_store = original_hp
    server_module._ds_store = original_ds
    server_module._launcher = original_launcher


class TestLaunchTraining:
    def test_launch_training(self, client, mock_launcher):
        resp = client.post(
            "/api/training/launch",
            json={
                "name": "Test Launch",
                "hp_pack_id": "hp-123",
                "dataset_id": "ds-456",
                "tags": ["test"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment_id"] == "test-launched-exp-id"
        assert data["status"] == "launched"

        mock_launcher.launch.assert_called_once_with(
            name="Test Launch",
            hp_pack_id="hp-123",
            dataset_id="ds-456",
            tags=["test"],
            overrides=None,
        )

    def test_launch_training_with_overrides(self, client, mock_launcher):
        resp = client.post(
            "/api/training/launch",
            json={
                "name": "Override Launch",
                "hp_pack_id": "hp-123",
                "dataset_id": "ds-456",
                "tags": [],
                "overrides": {"learning_rate": 1e-3},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "launched"

    def test_launch_training_missing_fields(self, client):
        resp = client.post(
            "/api/training/launch",
            json={
                "name": "Incomplete",
            },
        )
        data = resp.json()
        assert "error" in data

    def test_launch_training_value_error(self, client, mock_launcher):
        mock_launcher.launch.side_effect = ValueError("HP pack not found")
        resp = client.post(
            "/api/training/launch",
            json={
                "name": "Bad Launch",
                "hp_pack_id": "bad-hp",
                "dataset_id": "ds-456",
            },
        )
        data = resp.json()
        assert "error" in data
        assert "HP pack not found" in data["error"]


class TestListRunning:
    def test_list_running_empty(self, client, mock_launcher):
        resp = client.get("/api/training/running")
        assert resp.status_code == 200
        data = resp.json()
        assert data == []

    def test_list_running_with_experiments(self, client, mock_launcher):
        mock_launcher.list_running.return_value = [
            {
                "experiment_id": "exp-1",
                "name": "Running Exp",
                "started_at": "2024-01-01T00:00:00",
                "pid": 12345,
                "tags": ["test"],
            }
        ]
        resp = client.get("/api/training/running")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Running Exp"


class TestCancelTraining:
    def test_cancel_training_not_found(self, client, mock_launcher):
        resp = client.post("/api/training/nonexistent/cancel")
        data = resp.json()
        assert "error" in data

    def test_cancel_training_success(self, client, mock_launcher):
        mock_launcher.cancel.return_value = True
        resp = client.post("/api/training/exp-123/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        mock_launcher.cancel.assert_called_once_with("exp-123")


class TestLaunchCampaign:
    def test_launch_campaign_with_search_space(self, client, mock_launcher):
        search_space = {
            "trade_penalty_multiplier": {
                "mode": "tune",
                "type": "float",
                "low": 0.8,
                "high": 1.8,
            }
        }
        resp = client.post(
            "/api/campaign/launch",
            json={
                "study_name": "alpha_study",
                "dataset_id": "ds-1",
                "n_trials": 10,
                "iterations_per_trial": 20,
                "action_schemes": ["BSH"],
                "reward_schemes": ["PBR"],
                "search_space": search_space,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["study_name"] == "test-study"
        assert data["status"] == "launched"
        mock_launcher.launch_campaign.assert_called_once_with(
            study_name="alpha_study",
            dataset_id="ds-1",
            n_trials=10,
            iterations_per_trial=20,
            action_schemes=["BSH"],
            reward_schemes=["PBR"],
            search_space=search_space,
        )
