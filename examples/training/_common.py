"""
Shared helpers for training scripts.

Provides argument parsing, experiment store setup, TensorBoard
logger creation, and dashboard bridge initialization.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensortrade.training.experiment_store import ExperimentStore
    from tensortrade.training.tensorboard import TradingTensorBoardLogger
    from tensortrade.api.training_bridge import TrainingBridge


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common training intelligence flags to an argument parser."""
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging alongside console output",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable real-time dashboard (starts FastAPI server)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment in the store",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=[],
        help="Tags for this experiment (e.g. --tags baseline test)",
    )
    return parser


def create_training_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with common training flags already added."""
    parser = argparse.ArgumentParser(description=description)
    return add_training_args(parser)


def setup_experiment(
    args: argparse.Namespace,
    script_name: str,
    config: dict,
) -> tuple[
    ExperimentStore,
    str,
    TradingTensorBoardLogger | None,
    TrainingBridge | None,
]:
    """Set up experiment tracking from parsed args.

    Returns (store, experiment_id, tb_logger, bridge).
    """
    from tensortrade.training.experiment_store import ExperimentStore

    store = ExperimentStore()

    exp_name = args.experiment_name or script_name
    experiment_id = store.create_experiment(
        name=exp_name,
        script=script_name,
        config=config,
        tags=args.tags if args.tags else [],
    )

    # TensorBoard
    tb_logger: TradingTensorBoardLogger | None = None
    if args.tensorboard:
        try:
            from tensortrade.training.tensorboard import (
                TradingTensorBoardLogger,
                TensorBoardConfig,
            )

            tb_config = TensorBoardConfig(
                log_dir=os.path.expanduser(f"~/ray_results/tensortrade/{exp_name}"),
            )
            tb_logger = TradingTensorBoardLogger(tb_config)
            print(f"TensorBoard: logging to {tb_config.log_dir}")
        except ImportError:
            print("Warning: torch.utils.tensorboard not available, skipping TB logging")

    # Dashboard bridge
    bridge: TrainingBridge | None = None
    if args.dashboard:
        try:
            from tensortrade.api.training_bridge import TrainingBridge
            from tensortrade.api.server import create_app
            import threading
            import uvicorn

            # Check if server already running
            import socket
            server_running = False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                server_running = s.connect_ex(("localhost", 8000)) == 0

            if not server_running:
                app = create_app()

                def _run_server() -> None:
                    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

                server_thread = threading.Thread(target=_run_server, daemon=True, name="api-server")
                server_thread.start()
                import time
                time.sleep(1)  # Give server time to start
                print("Dashboard: started server at http://localhost:8000")
            else:
                print("Dashboard: using existing server at http://localhost:8000")

            bridge = TrainingBridge()
            bridge.start()
            bridge.send({"type": "experiment_start", "experiment_id": experiment_id, "name": exp_name})
            print("Dashboard: bridge connected")
        except ImportError as e:
            print(f"Warning: dashboard deps not available ({e}), skipping")

    return store, experiment_id, tb_logger, bridge


def build_composed_callbacks(
    base_cls: type,
    store: ExperimentStore,
    experiment_id: str,
    tb_logger: TradingTensorBoardLogger | None = None,
    bridge: TrainingBridge | None = None,
) -> type:
    """Build composed callback class from integrations."""
    from tensortrade.training.callbacks import make_training_callbacks

    return make_training_callbacks(
        base_cls=base_cls,
        tb_logger=tb_logger,
        experiment_store=store,
        experiment_id=experiment_id,
        dashboard_bridge=bridge,
    )


def log_training_iteration(
    result: dict,
    iteration: int,
    store: ExperimentStore,
    experiment_id: str,
    tb_logger: TradingTensorBoardLogger | None = None,
    bridge: TrainingBridge | None = None,
) -> None:
    """Log a training iteration to experiment store, TensorBoard, and dashboard.

    Call this in the training loop after each ``algo.train()`` call.
    Runs on the driver process only (not in Ray workers).
    """
    custom = result.get("env_runners", {}).get("custom_metrics", {})
    env_runners = result.get("env_runners", {})
    metrics = {
        "episode_return_mean": env_runners.get("episode_return_mean", 0),
        "pnl_mean": custom.get("pnl_mean", 0),
        "pnl_pct_mean": custom.get("pnl_pct_mean", 0),
        "net_worth_mean": custom.get("final_net_worth_mean", 0),
        "trade_count_mean": custom.get("trade_count_mean", 0),
        "hold_count_mean": custom.get("hold_count_mean", 0),
    }

    try:
        store.log_iteration(experiment_id, iteration, metrics)
    except Exception:
        pass

    if tb_logger:
        try:
            tb_logger.log_training_result(result, iteration)
            tb_logger.flush()
        except Exception:
            pass

    if bridge:
        try:
            bridge.send({
                "type": "training_update",
                "iteration": iteration,
                "episode_return_mean": env_runners.get("episode_return_mean", 0),
                "pnl_mean": custom.get("pnl_mean", 0),
                "pnl_pct_mean": custom.get("pnl_pct_mean", 0),
                "net_worth_mean": custom.get("final_net_worth_mean", 0),
                "trade_count_mean": custom.get("trade_count_mean", 0),
                "hold_count_mean": custom.get("hold_count_mean", 0),
            })
        except Exception:
            pass


def finish_experiment(
    store: ExperimentStore,
    experiment_id: str,
    status: str = "completed",
    final_metrics: dict | None = None,
    tb_logger: TradingTensorBoardLogger | None = None,
    bridge: TrainingBridge | None = None,
) -> None:
    """Clean up experiment tracking."""
    store.complete_experiment(experiment_id, status, final_metrics)

    if tb_logger:
        tb_logger.close()

    if bridge:
        bridge.send({"type": "experiment_end", "experiment_id": experiment_id, "status": status})
        bridge.stop()
