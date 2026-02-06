"""Tests for Dataset API endpoints."""

import pytest
from starlette.testclient import TestClient

from tensortrade.api.server import create_app
from tensortrade.training.dataset_store import DatasetStore, SEED_DATASETS
from tensortrade.training.feature_engine import FEATURE_CATALOG


@pytest.fixture
def stores(tmp_path):
    db_path = str(tmp_path / "test_datasets.db")

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
def client(stores):
    import tensortrade.api.server as server_module

    exp_store, hp_store, ds_store = stores
    original_store = server_module._store
    original_hp = server_module._hp_store
    original_ds = server_module._ds_store
    original_fe = server_module._feature_engine

    server_module._store = exp_store
    server_module._hp_store = hp_store
    server_module._ds_store = ds_store

    from tensortrade.training.feature_engine import FeatureEngine
    server_module._feature_engine = FeatureEngine()

    app = create_app()
    yield TestClient(app, raise_server_exceptions=False)

    server_module._store = original_store
    server_module._hp_store = original_hp
    server_module._ds_store = original_ds
    server_module._feature_engine = original_fe


class TestListDatasets:
    def test_list_datasets_returns_seeds(self, client):
        resp = client.get("/api/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(SEED_DATASETS)
        assert all("id" in d for d in data)
        assert all("name" in d for d in data)
        assert all("source_type" in d for d in data)


class TestCreateDataset:
    def test_create_dataset(self, client):
        resp = client.post("/api/datasets", json={
            "name": "New Dataset",
            "description": "Test dataset",
            "source_type": "synthetic",
            "source_config": {"base_price": 100},
            "features": [{"type": "rsi", "period": 14}],
            "split_config": {"train_pct": 0.8, "val_pct": 0.1, "test_pct": 0.1},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert data["name"] == "New Dataset"
        assert data["source_type"] == "synthetic"

    def test_create_dataset_missing_name(self, client):
        resp = client.post("/api/datasets", json={"description": "No name"})
        data = resp.json()
        assert "error" in data


class TestGetDataset:
    def test_get_dataset(self, client):
        datasets = client.get("/api/datasets").json()
        ds_id = datasets[0]["id"]

        resp = client.get(f"/api/datasets/{ds_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == ds_id

    def test_get_dataset_not_found(self, client):
        resp = client.get("/api/datasets/nonexistent-uuid")
        data = resp.json()
        assert "error" in data


class TestUpdateDataset:
    def test_update_dataset(self, client):
        create_resp = client.post("/api/datasets", json={
            "name": "To Update",
            "source_type": "csv_upload",
        })
        ds_id = create_resp.json()["id"]

        resp = client.put(f"/api/datasets/{ds_id}", json={
            "name": "Updated Dataset",
            "description": "Updated via API",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Dataset"
        assert data["description"] == "Updated via API"

    def test_update_dataset_not_found(self, client):
        resp = client.put("/api/datasets/nonexistent-uuid", json={"name": "No DS"})
        data = resp.json()
        assert "error" in data


class TestDeleteDataset:
    def test_delete_dataset(self, client):
        create_resp = client.post("/api/datasets", json={
            "name": "To Delete",
            "source_type": "csv_upload",
        })
        ds_id = create_resp.json()["id"]

        resp = client.delete(f"/api/datasets/{ds_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

        get_resp = client.get(f"/api/datasets/{ds_id}")
        assert "error" in get_resp.json()

    def test_delete_dataset_not_found(self, client):
        resp = client.delete("/api/datasets/nonexistent-uuid")
        data = resp.json()
        assert "error" in data


class TestFeaturesCatalog:
    def test_features_catalog(self, client):
        resp = client.get("/api/datasets/features")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(FEATURE_CATALOG)
        types = {entry["type"] for entry in data}
        assert "rsi" in types
        assert "returns" in types
        assert "bollinger_position" in types
