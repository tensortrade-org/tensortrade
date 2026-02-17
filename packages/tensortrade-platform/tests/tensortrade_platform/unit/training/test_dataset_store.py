"""Tests for DatasetStore."""

import sqlite3

import pytest

from tensortrade_platform.training.dataset_store import (
    SEED_DATASETS,
    DatasetConfig,
    DatasetStore,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_ds.db")
    s = DatasetStore(db_path=db_path)
    yield s
    s.close()


class TestSeedDatasets:
    def test_default_datasets_seeded(self, store):
        configs = store.list_configs()
        assert len(configs) == len(SEED_DATASETS)

    def test_seed_not_duplicated_on_reopen(self, tmp_path):
        db_path = str(tmp_path / "test_reseed.db")
        s1 = DatasetStore(db_path=db_path)
        assert len(s1.list_configs()) == len(SEED_DATASETS)
        s1.close()

        s2 = DatasetStore(db_path=db_path)
        assert len(s2.list_configs()) == len(SEED_DATASETS)
        s2.close()


class TestCreateConfig:
    def test_create_config(self, store):
        features = [{"type": "rsi", "period": 14}]
        split_config = {"train_pct": 0.8, "val_pct": 0.1, "test_pct": 0.1}
        ds_id = store.create_config(
            name="Test Dataset",
            description="A test dataset",
            source_type="synthetic",
            source_config={"base_price": 100},
            features=features,
            split_config=split_config,
        )
        assert isinstance(ds_id, str)
        assert len(ds_id) == 36

        config = store.get_config(ds_id)
        assert config is not None
        assert isinstance(config, DatasetConfig)
        assert config.name == "Test Dataset"
        assert config.description == "A test dataset"
        assert config.source_type == "synthetic"
        assert config.source_config == {"base_price": 100}
        assert config.features == features
        assert config.split_config == split_config

    def test_create_config_defaults(self, store):
        ds_id = store.create_config(name="Minimal")
        config = store.get_config(ds_id)
        assert config is not None
        assert config.source_type == "csv_upload"
        assert config.source_config == {}
        assert config.features == []
        assert config.split_config == {
            "train_pct": 0.7,
            "val_pct": 0.15,
            "test_pct": 0.15,
        }


class TestGetConfig:
    def test_get_config(self, store):
        ds_id = store.create_config(name="Findable", source_type="synthetic")
        config = store.get_config(ds_id)
        assert config is not None
        assert config.id == ds_id

    def test_get_config_not_found(self, store):
        result = store.get_config("nonexistent-uuid")
        assert result is None

    def test_get_config_by_name(self, store):
        store.create_config(name="Named Dataset")
        config = store.get_config_by_name("Named Dataset")
        assert config is not None
        assert config.name == "Named Dataset"

    def test_get_config_by_name_not_found(self, store):
        result = store.get_config_by_name("Does Not Exist")
        assert result is None


class TestListConfigs:
    def test_list_configs_with_seeds(self, store):
        configs = store.list_configs()
        assert len(configs) == len(SEED_DATASETS)
        assert all(isinstance(c, DatasetConfig) for c in configs)

    def test_list_configs_includes_created(self, store):
        store.create_config(name="ZZZZZ New Dataset")
        configs = store.list_configs()
        assert len(configs) == len(SEED_DATASETS) + 1
        names = [c.name for c in configs]
        assert "ZZZZZ New Dataset" in names


class TestUpdateConfig:
    def test_update_config_name(self, store):
        ds_id = store.create_config(name="Original Name")
        result = store.update_config(ds_id, name="New Name")
        assert result is True

        config = store.get_config(ds_id)
        assert config is not None
        assert config.name == "New Name"

    def test_update_config_description(self, store):
        ds_id = store.create_config(name="Desc Test")
        result = store.update_config(ds_id, description="Updated description")
        assert result is True

        config = store.get_config(ds_id)
        assert config is not None
        assert config.description == "Updated description"

    def test_update_config_features(self, store):
        ds_id = store.create_config(name="Feature Test")
        new_features = [{"type": "rsi", "period": 21}]
        result = store.update_config(ds_id, features=new_features)
        assert result is True

        config = store.get_config(ds_id)
        assert config is not None
        assert config.features == new_features

    def test_update_config_preserves_unchanged(self, store):
        ds_id = store.create_config(
            name="Preserve",
            description="Keep this",
            source_type="synthetic",
        )
        store.update_config(ds_id, name="New Name Only")

        config = store.get_config(ds_id)
        assert config is not None
        assert config.name == "New Name Only"
        assert config.description == "Keep this"
        assert config.source_type == "synthetic"

    def test_update_config_not_found(self, store):
        result = store.update_config("nonexistent-uuid", name="No Config")
        assert result is False

    def test_update_config_updates_timestamp(self, store):
        ds_id = store.create_config(name="Timestamp Test")
        before = store.get_config(ds_id)
        assert before is not None

        store.update_config(ds_id, name="Timestamp Updated")
        after = store.get_config(ds_id)
        assert after is not None
        assert after.updated_at >= before.updated_at


class TestDeleteConfig:
    def test_delete_config(self, store):
        ds_id = store.create_config(name="To Delete")
        result = store.delete_config(ds_id)
        assert result is True

        config = store.get_config(ds_id)
        assert config is None

    def test_delete_config_not_found(self, store):
        result = store.delete_config("nonexistent-uuid")
        assert result is False


class TestUniqueNameConstraint:
    def test_unique_name_constraint(self, store):
        store.create_config(name="Unique Name")
        with pytest.raises(sqlite3.IntegrityError):
            store.create_config(name="Unique Name")


class TestAliases:
    def test_get_dataset_alias(self, store):
        ds_id = store.create_config(name="Alias Test")
        config = store.get_dataset(ds_id)
        assert config is not None
        assert config.name == "Alias Test"

    def test_list_datasets_alias(self, store):
        datasets = store.list_datasets()
        assert len(datasets) == len(SEED_DATASETS)

    def test_create_dataset_alias(self, store):
        config = store.create_dataset(name="Alias Create")
        assert isinstance(config, DatasetConfig)
        assert config.name == "Alias Create"

    def test_update_dataset_alias(self, store):
        ds_id = store.create_config(name="Alias Update")
        result = store.update_dataset(ds_id, name="Updated Via Alias")
        assert result is not None
        assert result.name == "Updated Via Alias"

    def test_update_dataset_alias_not_found(self, store):
        result = store.update_dataset("nonexistent-uuid", name="No Config")
        assert result is None

    def test_delete_dataset_alias(self, store):
        ds_id = store.create_config(name="Alias Delete")
        result = store.delete_dataset(ds_id)
        assert result is True


class TestDBLifecycle:
    def test_data_persists_across_connections(self, tmp_path):
        db_path = str(tmp_path / "persist_ds.db")
        s1 = DatasetStore(db_path=db_path)
        ds_id = s1.create_config(name="Persistent")
        s1.close()

        s2 = DatasetStore(db_path=db_path)
        config = s2.get_config(ds_id)
        assert config is not None
        assert config.name == "Persistent"
        s2.close()
