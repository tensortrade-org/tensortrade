"""Tests for TrainingLauncher."""

from unittest.mock import MagicMock

import pytest

from tensortrade.training.dataset_store import DatasetConfig
from tensortrade.training.hyperparameter_store import HyperparameterPack
from tensortrade.training.launcher import TrainingLauncher


def _make_hp_pack(**overrides) -> HyperparameterPack:
    defaults = {
        "id": "hp-1",
        "name": "Test Pack",
        "description": "A test pack",
        "config": {"algorithm": "PPO", "learning_rate": 5e-5, "gamma": 0.99},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    defaults.update(overrides)
    return HyperparameterPack(**defaults)


def _make_dataset_config(**overrides) -> DatasetConfig:
    defaults = {
        "id": "ds-1",
        "name": "Test Dataset",
        "description": "A test dataset",
        "source_type": "synthetic",
        "source_config": {"base_price": 50000, "num_candles": 1000},
        "features": [{"type": "rsi", "period": 14}],
        "split_config": {"train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    defaults.update(overrides)
    return DatasetConfig(**defaults)


@pytest.fixture
def mock_stores():
    experiment_store = MagicMock()
    experiment_store.create_experiment.return_value = "test-exp-id"
    hp_store = MagicMock()
    ds_store = MagicMock()
    return experiment_store, hp_store, ds_store


@pytest.fixture
def launcher(mock_stores):
    experiment_store, hp_store, ds_store = mock_stores
    return TrainingLauncher(experiment_store, hp_store, ds_store)


class TestLaunchResolvesConfig:
    def test_launch_resolves_hp_pack(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = _make_hp_pack()
        ds_store.get_config.return_value = _make_dataset_config()

        # Mock subprocess to avoid actually spawning processes
        from unittest.mock import patch

        mock_process = MagicMock()
        mock_process.pid = 12345
        with patch("tensortrade.training.launcher.subprocess.Popen", return_value=mock_process):
            exp_id = launcher.launch(
                name="Test Run",
                hp_pack_id="hp-1",
                dataset_id="ds-1",
                tags=["test"],
            )

        assert exp_id == "test-exp-id"
        hp_store.get_pack.assert_called_once_with("hp-1")
        ds_store.get_config.assert_called_once_with("ds-1")

    def test_launch_merges_overrides(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = _make_hp_pack(config={"algorithm": "PPO", "learning_rate": 5e-5})
        ds_store.get_config.return_value = _make_dataset_config()

        from unittest.mock import patch

        mock_process = MagicMock()
        mock_process.pid = 12345
        with patch("tensortrade.training.launcher.subprocess.Popen", return_value=mock_process):
            launcher.launch(
                name="Override Run",
                hp_pack_id="hp-1",
                dataset_id="ds-1",
                overrides={"learning_rate": 1e-4},
            )

        # Verify the experiment was created with merged config
        call_kwargs = mock_stores[0].create_experiment.call_args[1]
        training_config = call_kwargs["config"]["training_config"]
        assert training_config["learning_rate"] == 1e-4
        assert training_config["algorithm"] == "PPO"


class TestLaunchErrors:
    def test_launch_hp_pack_not_found(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = None

        with pytest.raises(ValueError, match="Hyperparameter pack not found"):
            launcher.launch(
                name="Bad Run",
                hp_pack_id="nonexistent",
                dataset_id="ds-1",
            )

    def test_launch_dataset_not_found(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = _make_hp_pack()
        ds_store.get_config.return_value = None

        with pytest.raises(ValueError, match="Dataset config not found"):
            launcher.launch(
                name="Bad Run",
                hp_pack_id="hp-1",
                dataset_id="nonexistent",
            )


class TestListRunning:
    def test_list_running_empty(self, launcher):
        result = launcher.list_running()
        assert result == []

    def test_list_running_with_process(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = _make_hp_pack()
        ds_store.get_config.return_value = _make_dataset_config()

        from unittest.mock import patch

        mock_process = MagicMock()
        mock_process.pid = 99999
        mock_process.poll.return_value = None  # Still running

        with patch("tensortrade.training.launcher.subprocess.Popen", return_value=mock_process):
            launcher.launch(
                name="Running Experiment",
                hp_pack_id="hp-1",
                dataset_id="ds-1",
            )

        running = launcher.list_running()
        assert len(running) == 1
        assert running[0]["name"] == "Running Experiment"
        assert running[0]["pid"] == 99999


class TestCancel:
    def test_cancel_not_found(self, launcher):
        result = launcher.cancel("nonexistent-id")
        assert result is False

    def test_cancel_running_experiment(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = _make_hp_pack()
        ds_store.get_config.return_value = _make_dataset_config()

        from unittest.mock import patch

        mock_process = MagicMock()
        mock_process.pid = 99999
        mock_process.poll.return_value = None

        with patch("tensortrade.training.launcher.subprocess.Popen", return_value=mock_process):
            exp_id = launcher.launch(
                name="To Cancel",
                hp_pack_id="hp-1",
                dataset_id="ds-1",
            )

        with patch("os.killpg"):
            result = launcher.cancel(exp_id)

        assert result is True
        mock_stores[0].complete_experiment.assert_called_once()
        assert launcher.list_running() == []


class TestCleanupFinished:
    def test_cleanup_removes_finished(self, launcher, mock_stores):
        _, hp_store, ds_store = mock_stores
        hp_store.get_pack.return_value = _make_hp_pack()
        ds_store.get_config.return_value = _make_dataset_config()

        from unittest.mock import patch

        mock_process = MagicMock()
        mock_process.pid = 11111
        # First poll returns None (running), then returns 0 (finished)
        mock_process.poll.side_effect = [None, 0]

        with patch("tensortrade.training.launcher.subprocess.Popen", return_value=mock_process):
            launcher.launch(
                name="Will Finish",
                hp_pack_id="hp-1",
                dataset_id="ds-1",
            )

        # First list_running: process is still running
        running1 = launcher.list_running()
        assert len(running1) == 1

        # Second list_running: process has finished, should be cleaned up
        running2 = launcher.list_running()
        assert len(running2) == 0
