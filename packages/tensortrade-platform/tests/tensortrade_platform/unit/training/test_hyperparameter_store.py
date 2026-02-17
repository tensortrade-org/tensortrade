"""Tests for HyperparameterStore."""

import sqlite3

import pytest

from tensortrade_platform.training.hyperparameter_store import (
    SEED_PACKS,
    HyperparameterPack,
    HyperparameterStore,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_hp.db")
    s = HyperparameterStore(db_path=db_path)
    yield s
    s.close()


class TestSeedPacks:
    def test_default_packs_seeded(self, store):
        packs = store.list_packs()
        assert len(packs) == len(SEED_PACKS)

    def test_seed_packs_not_duplicated_on_reopen(self, tmp_path):
        db_path = str(tmp_path / "test_reseed.db")
        s1 = HyperparameterStore(db_path=db_path)
        assert len(s1.list_packs()) == len(SEED_PACKS)
        s1.close()

        s2 = HyperparameterStore(db_path=db_path)
        assert len(s2.list_packs()) == len(SEED_PACKS)
        s2.close()


class TestCreatePack:
    def test_create_pack(self, store):
        config = {"algorithm": "PPO", "learning_rate": 1e-4}
        pack_id = store.create_pack(
            name="Custom Pack", description="A test pack", config=config
        )
        assert isinstance(pack_id, str)
        assert len(pack_id) == 36

        pack = store.get_pack(pack_id)
        assert pack is not None
        assert isinstance(pack, HyperparameterPack)
        assert pack.name == "Custom Pack"
        assert pack.description == "A test pack"
        assert pack.config == config
        assert pack.created_at is not None
        assert pack.updated_at is not None

    def test_create_pack_no_config(self, store):
        pack_id = store.create_pack(name="Empty Config")
        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.config == {}

    def test_create_pack_no_description(self, store):
        pack_id = store.create_pack(name="No Desc", config={"lr": 0.01})
        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.description == ""


class TestGetPack:
    def test_get_pack(self, store):
        pack_id = store.create_pack(name="Findable", config={"gamma": 0.99})
        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.id == pack_id
        assert pack.name == "Findable"

    def test_get_pack_not_found(self, store):
        result = store.get_pack("nonexistent-uuid")
        assert result is None

    def test_get_pack_by_name(self, store):
        store.create_pack(name="Named Pack", config={"lr": 0.001})
        pack = store.get_pack_by_name("Named Pack")
        assert pack is not None
        assert pack.name == "Named Pack"

    def test_get_pack_by_name_not_found(self, store):
        result = store.get_pack_by_name("Does Not Exist")
        assert result is None


class TestListPacks:
    def test_list_packs(self, store):
        packs = store.list_packs()
        assert len(packs) == len(SEED_PACKS)
        assert all(isinstance(p, HyperparameterPack) for p in packs)

    def test_list_packs_includes_created(self, store):
        store.create_pack(name="ZZZZZ New Pack", config={"lr": 0.01})
        packs = store.list_packs()
        assert len(packs) == len(SEED_PACKS) + 1
        names = [p.name for p in packs]
        assert "ZZZZZ New Pack" in names


class TestUpdatePack:
    def test_update_pack_name(self, store):
        pack_id = store.create_pack(name="Original", config={"lr": 0.001})
        result = store.update_pack(pack_id, name="Updated Name")
        assert result is True

        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.name == "Updated Name"

    def test_update_pack_description(self, store):
        pack_id = store.create_pack(name="Desc Test")
        result = store.update_pack(pack_id, description="New description")
        assert result is True

        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.description == "New description"

    def test_update_pack_config(self, store):
        pack_id = store.create_pack(name="Config Test", config={"lr": 0.001})
        new_config = {"lr": 0.01, "gamma": 0.95}
        result = store.update_pack(pack_id, config=new_config)
        assert result is True

        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.config == new_config

    def test_update_pack_preserves_unchanged_fields(self, store):
        pack_id = store.create_pack(
            name="Preserve", description="Keep this", config={"lr": 0.001}
        )
        store.update_pack(pack_id, name="New Name Only")

        pack = store.get_pack(pack_id)
        assert pack is not None
        assert pack.name == "New Name Only"
        assert pack.description == "Keep this"
        assert pack.config == {"lr": 0.001}

    def test_update_pack_not_found(self, store):
        result = store.update_pack("nonexistent-uuid", name="No Pack")
        assert result is False

    def test_update_pack_updates_timestamp(self, store):
        pack_id = store.create_pack(name="Timestamp Test")
        pack_before = store.get_pack(pack_id)
        assert pack_before is not None

        store.update_pack(pack_id, name="Timestamp Updated")
        pack_after = store.get_pack(pack_id)
        assert pack_after is not None
        assert pack_after.updated_at >= pack_before.updated_at


class TestDeletePack:
    def test_delete_pack(self, store):
        pack_id = store.create_pack(name="To Delete", config={"lr": 0.01})
        result = store.delete_pack(pack_id)
        assert result is True

        pack = store.get_pack(pack_id)
        assert pack is None

    def test_delete_pack_not_found(self, store):
        result = store.delete_pack("nonexistent-uuid")
        assert result is False


class TestDuplicatePack:
    def test_duplicate_pack(self, store):
        original_id = store.create_pack(
            name="Original", description="Source pack", config={"lr": 0.001}
        )
        new_id = store.duplicate_pack(original_id)
        assert new_id != original_id

        new_pack = store.get_pack(new_id)
        assert new_pack is not None
        assert new_pack.name == "Original (copy)"
        assert new_pack.description == "Source pack"
        assert new_pack.config == {"lr": 0.001}

    def test_duplicate_pack_custom_name(self, store):
        original_id = store.create_pack(name="Source", config={"lr": 0.01})
        new_id = store.duplicate_pack(original_id, new_name="My Copy")

        new_pack = store.get_pack(new_id)
        assert new_pack is not None
        assert new_pack.name == "My Copy"

    def test_duplicate_pack_not_found(self, store):
        with pytest.raises(ValueError, match="Pack not found"):
            store.duplicate_pack("nonexistent-uuid")

    def test_duplicate_pack_name_conflict_resolved(self, store):
        original_id = store.create_pack(name="Dup Source", config={"lr": 0.001})
        store.create_pack(name="Dup Source (copy)", config={})

        new_id = store.duplicate_pack(original_id)
        new_pack = store.get_pack(new_id)
        assert new_pack is not None
        assert new_pack.name == "Dup Source (copy) (2)"


class TestUniqueNameConstraint:
    def test_unique_name_constraint(self, store):
        store.create_pack(name="Unique Name", config={"lr": 0.01})
        with pytest.raises(sqlite3.IntegrityError):
            store.create_pack(name="Unique Name", config={"lr": 0.02})


class TestDBLifecycle:
    def test_data_persists_across_connections(self, tmp_path):
        db_path = str(tmp_path / "persist_hp.db")
        s1 = HyperparameterStore(db_path=db_path)
        pack_id = s1.create_pack(name="Persistent", config={"lr": 0.001})
        s1.close()

        s2 = HyperparameterStore(db_path=db_path)
        pack = s2.get_pack(pack_id)
        assert pack is not None
        assert pack.name == "Persistent"
        s2.close()
