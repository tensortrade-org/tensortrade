"""Tests for HP pack API endpoints."""

import pytest
from starlette.testclient import TestClient

from tensortrade.api.server import create_app
from tensortrade.training.hyperparameter_store import SEED_PACKS, HyperparameterStore


@pytest.fixture
def stores(tmp_path):
    db_path = str(tmp_path / "test_packs.db")
    hp_store = HyperparameterStore(db_path=db_path)

    from tensortrade.training.dataset_store import DatasetStore
    from tensortrade.training.experiment_store import ExperimentStore

    exp_store = ExperimentStore(db_path=db_path)
    ds_store = DatasetStore(db_path=db_path)
    yield exp_store, hp_store, ds_store
    hp_store.close()
    exp_store.close()
    ds_store.close()


@pytest.fixture
def client(stores):
    import tensortrade.api.server as server_module

    exp_store, hp_store, ds_store = stores
    original_store = server_module._store
    original_hp = server_module._hp_store
    original_ds = server_module._ds_store

    server_module._store = exp_store
    server_module._hp_store = hp_store
    server_module._ds_store = ds_store

    app = create_app()
    yield TestClient(app, raise_server_exceptions=False)

    server_module._store = original_store
    server_module._hp_store = original_hp
    server_module._ds_store = original_ds


class TestListPacks:
    def test_list_packs_returns_defaults(self, client):
        resp = client.get("/api/packs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == len(SEED_PACKS)
        assert all("id" in p for p in data)
        assert all("name" in p for p in data)
        assert all("config" in p for p in data)


class TestCreatePack:
    def test_create_pack(self, client):
        resp = client.post(
            "/api/packs",
            json={
                "name": "New Custom Pack",
                "description": "Created via API",
                "config": {"algorithm": "PPO", "learning_rate": 1e-4},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert data["name"] == "New Custom Pack"
        assert data["config"]["algorithm"] == "PPO"

    def test_create_pack_missing_name(self, client):
        resp = client.post("/api/packs", json={"description": "No name"})
        data = resp.json()
        assert "error" in data


class TestGetPack:
    def test_get_pack(self, client):
        # Get the first default pack
        packs = client.get("/api/packs").json()
        pack_id = packs[0]["id"]

        resp = client.get(f"/api/packs/{pack_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == pack_id

    def test_get_pack_not_found(self, client):
        resp = client.get("/api/packs/nonexistent-uuid")
        data = resp.json()
        assert "error" in data


class TestUpdatePack:
    def test_update_pack(self, client):
        # Create a pack first
        create_resp = client.post(
            "/api/packs",
            json={
                "name": "To Update",
                "config": {"lr": 0.001},
            },
        )
        pack_id = create_resp.json()["id"]

        resp = client.put(
            f"/api/packs/{pack_id}",
            json={
                "name": "Updated Pack",
                "description": "Updated description",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Pack"
        assert data["description"] == "Updated description"

    def test_update_pack_not_found(self, client):
        resp = client.put("/api/packs/nonexistent-uuid", json={"name": "No Pack"})
        data = resp.json()
        assert "error" in data


class TestDeletePack:
    def test_delete_pack(self, client):
        create_resp = client.post(
            "/api/packs",
            json={
                "name": "To Delete",
                "config": {},
            },
        )
        pack_id = create_resp.json()["id"]

        resp = client.delete(f"/api/packs/{pack_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

        # Verify it's gone
        get_resp = client.get(f"/api/packs/{pack_id}")
        assert "error" in get_resp.json()

    def test_delete_pack_not_found(self, client):
        resp = client.delete("/api/packs/nonexistent-uuid")
        data = resp.json()
        assert "error" in data


class TestDuplicatePack:
    def test_duplicate_pack(self, client):
        packs = client.get("/api/packs").json()
        pack_id = packs[0]["id"]
        original_name = packs[0]["name"]

        resp = client.post(f"/api/packs/{pack_id}/duplicate")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert data["name"] == f"{original_name} (copy)"
        assert data["id"] != pack_id

    def test_duplicate_pack_not_found(self, client):
        resp = client.post("/api/packs/nonexistent-uuid/duplicate")
        data = resp.json()
        assert "error" in data
