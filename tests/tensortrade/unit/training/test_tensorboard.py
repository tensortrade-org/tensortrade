"""Tests for the TensorBoard logger."""

import os
import pytest
from unittest.mock import MagicMock, patch, call


class TestTensorBoardConfig:
    def test_defaults(self):
        from tensortrade.training.tensorboard import TensorBoardConfig

        cfg = TensorBoardConfig()
        assert cfg.log_dir == "~/ray_results/tensortrade"
        assert cfg.flush_secs == 30

    def test_custom_values(self):
        from tensortrade.training.tensorboard import TensorBoardConfig

        cfg = TensorBoardConfig(log_dir="/tmp/tb", flush_secs=10)
        assert cfg.log_dir == "/tmp/tb"
        assert cfg.flush_secs == 10


class TestTradingTensorBoardLogger:
    @patch("tensortrade.training.tensorboard.TradingTensorBoardLogger._setup_layout")
    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_training_result(self, mock_writer_cls, mock_layout, tmp_path):
        from tensortrade.training.tensorboard import (
            TradingTensorBoardLogger,
            TensorBoardConfig,
        )

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        config = TensorBoardConfig(log_dir=str(tmp_path / "tb"))
        logger = TradingTensorBoardLogger(config)

        result = {
            "env_runners": {
                "episode_return_mean": 150.0,
                "custom_metrics": {
                    "pnl_mean": 50.0,
                    "pnl_pct_mean": 5.0,
                    "final_net_worth_mean": 10500.0,
                    "trade_count_mean": 12,
                    "hold_count_mean": 88,
                },
            },
            "learners": {
                "default_policy": {
                    "total_loss": 0.5,
                    "policy_loss": 0.3,
                    "vf_loss": 0.2,
                },
            },
        }
        logger.log_training_result(result, iteration=1)

        # Check that expected scalars were logged
        scalar_calls = mock_writer.add_scalar.call_args_list
        tags_logged = {c[0][0] for c in scalar_calls}

        assert "Trading/pnl" in tags_logged
        assert "Trading/pnl_pct" in tags_logged
        assert "Trading/net_worth" in tags_logged
        assert "Behavior/trade_count" in tags_logged
        assert "Behavior/hold_count" in tags_logged
        assert "Performance/episode_return_mean" in tags_logged
        assert "Performance/total_loss" in tags_logged

    @patch("tensortrade.training.tensorboard.TradingTensorBoardLogger._setup_layout")
    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_evaluation(self, mock_writer_cls, mock_layout, tmp_path):
        from tensortrade.training.tensorboard import (
            TradingTensorBoardLogger,
            TensorBoardConfig,
        )

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        config = TensorBoardConfig(log_dir=str(tmp_path / "tb"))
        logger = TradingTensorBoardLogger(config)

        logger.log_evaluation(
            {"pnl": 100.0, "win_rate": 0.6, "text_field": "ignore"},
            iteration=5,
            prefix="Val",
        )

        scalar_calls = mock_writer.add_scalar.call_args_list
        tags_logged = {c[0][0] for c in scalar_calls}
        assert "Val/pnl" in tags_logged
        assert "Val/win_rate" in tags_logged
        # Non-numeric fields should not be logged
        assert "Val/text_field" not in tags_logged

    @patch("tensortrade.training.tensorboard.TradingTensorBoardLogger._setup_layout")
    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_flush_and_close(self, mock_writer_cls, mock_layout, tmp_path):
        from tensortrade.training.tensorboard import (
            TradingTensorBoardLogger,
            TensorBoardConfig,
        )

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        config = TensorBoardConfig(log_dir=str(tmp_path / "tb"))
        logger = TradingTensorBoardLogger(config)

        logger.flush()
        mock_writer.flush.assert_called_once()

        logger.close()
        mock_writer.close.assert_called_once()

    @patch("tensortrade.training.tensorboard.TradingTensorBoardLogger._setup_layout")
    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_handles_empty_result(self, mock_writer_cls, mock_layout, tmp_path):
        from tensortrade.training.tensorboard import (
            TradingTensorBoardLogger,
            TensorBoardConfig,
        )

        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        config = TensorBoardConfig(log_dir=str(tmp_path / "tb"))
        logger = TradingTensorBoardLogger(config)

        # Empty result dict should not crash
        logger.log_training_result({}, iteration=0)
        # Should still log defaults (0 values)
        assert mock_writer.add_scalar.called
