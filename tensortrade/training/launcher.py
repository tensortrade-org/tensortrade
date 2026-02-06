"""
Training launcher service.

Spawns training runs as subprocesses from the API, connecting them
to the dashboard via the training bridge for live progress streaming.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensortrade.training.experiment_store import ExperimentStore
    from tensortrade.training.hyperparameter_store import HyperparameterStore
    from tensortrade.training.dataset_store import DatasetStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


@dataclass
class RunningExperiment:
    experiment_id: str
    name: str
    process: subprocess.Popen | None
    pid: int | None
    started_at: str
    config: dict
    hp_pack_id: str | None = None
    dataset_id: str | None = None
    tags: list[str] = field(default_factory=list)


class TrainingLauncher:
    """Service to spawn and manage training subprocess runs."""

    def __init__(
        self,
        experiment_store: ExperimentStore,
        hp_store: HyperparameterStore,
        dataset_store: DatasetStore,
    ) -> None:
        self._experiment_store = experiment_store
        self._hp_store = hp_store
        self._dataset_store = dataset_store
        self._running: dict[str, RunningExperiment] = {}

    def launch(
        self,
        name: str,
        hp_pack_id: str,
        dataset_id: str,
        tags: list[str] | None = None,
        overrides: dict | None = None,
    ) -> str:
        """Launch a training run as a subprocess.

        Only one training run may be active at a time.
        Returns the experiment_id.
        """
        self._cleanup_finished()
        if self._running:
            raise RuntimeError(
                "A training run is already active. Cancel it before starting a new one."
            )

        # Resolve HP pack
        hp_pack = self._hp_store.get_pack(hp_pack_id)
        if hp_pack is None:
            raise ValueError(f"Hyperparameter pack not found: {hp_pack_id}")

        # Resolve dataset config
        dataset = self._dataset_store.get_config(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset config not found: {dataset_id}")

        # Build merged config
        config = dict(hp_pack.config)
        if overrides:
            config.update(overrides)

        # Create experiment in store
        experiment_id = self._experiment_store.create_experiment(
            name=name,
            script="ui_launch",
            config={
                "training_config": config,
                "hp_pack_id": hp_pack_id,
                "hp_pack_name": hp_pack.name,
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "source_type": dataset.source_type,
                "source_config": dataset.source_config,
                "features": dataset.features,
                "split_config": dataset.split_config,
            },
            tags=tags or [],
        )

        # Generate and write the training script
        script_path = self._generate_script(experiment_id, name, config, dataset)

        # Spawn subprocess
        python = sys.executable
        process = subprocess.Popen(
            [python, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        now = datetime.now(timezone.utc).isoformat()
        running = RunningExperiment(
            experiment_id=experiment_id,
            name=name,
            process=process,
            pid=process.pid,
            started_at=now,
            config=config,
            hp_pack_id=hp_pack_id,
            dataset_id=dataset_id,
            tags=tags or [],
        )
        self._running[experiment_id] = running

        logger.info(
            "Launched training %s (pid=%s, experiment=%s)",
            name,
            process.pid,
            experiment_id,
        )
        return experiment_id

    def list_running(self) -> list[dict]:
        """List currently running experiments."""
        self._cleanup_finished()
        result: list[dict] = []
        for exp_id, running in self._running.items():
            result.append({
                "experiment_id": exp_id,
                "name": running.name,
                "started_at": running.started_at,
                "pid": running.pid,
                "tags": running.tags,
            })
        return result

    def cancel(self, experiment_id: str) -> bool:
        """Cancel a running experiment. Returns True if cancelled."""
        running = self._running.get(experiment_id)
        if running is None:
            return False

        if running.process is not None:
            try:
                os.killpg(os.getpgid(running.process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        self._experiment_store.complete_experiment(
            experiment_id, status="failed", final_metrics={"cancelled": True}
        )
        del self._running[experiment_id]
        logger.info("Cancelled training %s", experiment_id)
        return True

    def _cleanup_finished(self) -> None:
        """Remove finished processes from the running dict."""
        finished: list[str] = []
        for exp_id, running in self._running.items():
            if running.process is not None and running.process.poll() is not None:
                finished.append(exp_id)
        for exp_id in finished:
            del self._running[exp_id]

    def _generate_script(
        self,
        experiment_id: str,
        name: str,
        config: dict,
        dataset: object,
    ) -> str:
        """Generate a temporary training script and return its path."""
        from tensortrade.training.dataset_store import DatasetConfig

        assert isinstance(dataset, DatasetConfig)

        tmp_dir = os.path.expanduser("~/.tensortrade/launch_scripts")
        os.makedirs(tmp_dir, exist_ok=True)
        script_path = os.path.join(tmp_dir, f"launch_{experiment_id}.py")

        config_json = json.dumps(config)
        dataset_source_json = json.dumps(dataset.source_config)
        features_json = json.dumps(dataset.features)
        split_json = json.dumps(dataset.split_config)

        lines = [
            '#!/usr/bin/env python3',
            f'"""Auto-generated training script for experiment {experiment_id}."""',
            '',
            'import json',
            'import os',
            'import sys',
            'import time',
            '',
            '# Ensure project root is in path',
            f'sys.path.insert(0, "{PROJECT_ROOT}")',
            '',
            'import numpy as np',
            'import pandas as pd',
            '',
            'import ray',
            'from ray.tune.registry import register_env',
            'from ray.rllib.algorithms.ppo import PPOConfig',
            'from ray.rllib.algorithms.callbacks import DefaultCallbacks',
            '',
            'from tensortrade.data.cdd import CryptoDataDownload',
            'from tensortrade.feed.core import DataFeed, Stream',
            'from tensortrade.oms.exchanges import Exchange, ExchangeOptions',
            'from tensortrade.oms.instruments import USD, BTC',
            'from tensortrade.oms.services.execution.simulated import execute_order',
            'from tensortrade.oms.wallets import Wallet, Portfolio',
            'from tensortrade.env.default.actions import BSH',
            'from tensortrade.env.default.rewards import PBR',
            'import tensortrade.env.default as default',
            'from tensortrade.training.experiment_store import ExperimentStore',
            'from tensortrade.training.feature_engine import FeatureEngine',
            '',
            f'EXPERIMENT_ID = "{experiment_id}"',
            f"CONFIG = json.loads('{config_json}')",
            f'SOURCE_TYPE = "{dataset.source_type}"',
            f"SOURCE_CONFIG = json.loads('{dataset_source_json}')",
            f"FEATURES = json.loads('{features_json}')",
            f"SPLIT_CONFIG = json.loads('{split_json}')",
            '',
            '',
            'class Callbacks(DefaultCallbacks):',
            '    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):',
            '        env = base_env.get_sub_environments()[env_index]',
            '        if hasattr(env, "portfolio"):',
            '            episode.user_data["initial"] = float(env.portfolio.net_worth)',
            '        if hasattr(env, "_action_counts"):',
            '            env._action_counts = {0: 0, 1: 0, 2: 0}',
            '',
            '    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):',
            '        env = base_env.get_sub_environments()[env_index]',
            '        if hasattr(env, "portfolio"):',
            '            final = float(env.portfolio.net_worth)',
            '            initial = episode.user_data.get("initial", 10000)',
            '            episode.custom_metrics["pnl"] = final - initial',
            '            episode.custom_metrics["pnl_pct"] = (final - initial) / initial * 100',
            '            episode.custom_metrics["final_net_worth"] = final',
            '        actions = getattr(env, "_action_counts", {0: 0, 1: 0, 2: 0})',
            '        episode.custom_metrics["hold_count"] = actions.get(0, 0)',
            '        episode.custom_metrics["buy_count"] = actions.get(1, 0)',
            '        episode.custom_metrics["sell_count"] = actions.get(2, 0)',
            '        episode.custom_metrics["trade_count"] = actions.get(1, 0) + actions.get(2, 0)',
            '',
            '',
            'def load_data():',
            '    if SOURCE_TYPE == "crypto_download":',
            '        cdd = CryptoDataDownload()',
            '        data = cdd.fetch(',
            '            SOURCE_CONFIG["exchange"],',
            '            SOURCE_CONFIG["base"],',
            '            SOURCE_CONFIG["quote"],',
            '            SOURCE_CONFIG["timeframe"],',
            '        )',
            '        data = data[["date", "open", "high", "low", "close", "volume"]]',
            '        data["date"] = pd.to_datetime(data["date"])',
            '        data.sort_values("date", inplace=True)',
            '        data.reset_index(drop=True, inplace=True)',
            '    elif SOURCE_TYPE == "synthetic":',
            '        from tensortrade.stochastic.processes.gbm import gbm',
            '        data = gbm(',
            '            base_price=SOURCE_CONFIG.get("base_price", 50000),',
            '            base_volume=SOURCE_CONFIG.get("base_volume", 1000),',
            '            start_date="2020-01-01",',
            '            times_to_generate=SOURCE_CONFIG.get("num_candles", 5000),',
            '            time_frame=SOURCE_CONFIG.get("timeframe", "1h"),',
            '        )',
            '        data.index.name = "date"',
            '        data = data.reset_index()',
            '        data["date"] = pd.to_datetime(data["date"])',
            '    else:',
            '        raise ValueError(f"Unsupported source_type: {SOURCE_TYPE}")',
            '',
            '    engine = FeatureEngine()',
            '    data = engine.compute(data, FEATURES)',
            '    return data',
            '',
            '',
            'def create_env(env_config):',
            '    data = pd.read_csv(env_config["csv_filename"]).bfill().ffill()',
            '    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")',
            '',
            '    commission = env_config.get("commission", CONFIG.get("commission", 0.003))',
            '    exchange = Exchange("exchange", service=execute_order,',
            '                       options=ExchangeOptions(commission=commission))(price)',
            '',
            '    cash = Wallet(exchange, env_config.get("initial_cash", CONFIG.get("initial_cash", 10000)) * USD)',
            '    asset = Wallet(exchange, 0 * BTC)',
            '    portfolio = Portfolio(USD, [cash, asset])',
            '',
            '    features = [Stream.source(list(data[c]), dtype="float").rename(c)',
            '               for c in env_config.get("feature_cols", [])]',
            '    feed = DataFeed(features)',
            '    feed.compile()',
            '',
            '    reward_scheme = PBR(price=price)',
            '    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)',
            '',
            '    max_ep = env_config.get("max_episode_steps") or CONFIG.get("max_episode_steps")',
            '    env = default.create(',
            '        feed=feed,',
            '        portfolio=portfolio,',
            '        action_scheme=action_scheme,',
            '        reward_scheme=reward_scheme,',
            '        window_size=env_config.get("window_size", CONFIG.get("window_size", 10)),',
            '        max_allowed_loss=env_config.get("max_allowed_loss", CONFIG.get("max_allowed_loss", 0.4)),',
            '        max_episode_steps=int(max_ep) if max_ep else None,',
            '        random_start_pct=env_config.get("random_start_pct", 0.5),',
            '    )',
            '    env.portfolio = portfolio',
            '    # Track actions for reporting',
            '    env._action_counts = {0: 0, 1: 0, 2: 0}',
            '    _orig_step = env.step',
            '    def _tracking_step(action):',
            '        a = int(action)',
            '        env._action_counts[a] = env._action_counts.get(a, 0) + 1',
            '        return _orig_step(action)',
            '    env.step = _tracking_step',
            '    return env',
            '',
            '',
            'def main():',
            '    store = ExperimentStore()',
            '',
            '    # Load and prepare data',
            '    data = load_data()',
            '    engine = FeatureEngine()',
            '    feature_cols = engine.get_feature_columns(FEATURES)',
            '',
            '    # Split data',
            '    train_pct = SPLIT_CONFIG.get("train_pct", 0.7)',
            '    val_pct = SPLIT_CONFIG.get("val_pct", 0.15)',
            '    n = len(data)',
            '    train_end = int(n * train_pct)',
            '',
            '    train_data = data.iloc[:train_end].reset_index(drop=True)',
            '    csv_path = os.path.expanduser(f"~/.tensortrade/launch_data_{EXPERIMENT_ID}.csv")',
            '    train_data.to_csv(csv_path, index=False)',
            '',
            '    # Default max_episode_steps to 500 if not set, so episodes complete',
            '    max_ep_steps = CONFIG.get("max_episode_steps") or 500',
            '    env_config = {',
            '        "csv_filename": csv_path,',
            '        "feature_cols": feature_cols,',
            '        "window_size": CONFIG.get("window_size", 10),',
            '        "max_allowed_loss": CONFIG.get("max_allowed_loss", 0.4),',
            '        "max_episode_steps": int(max_ep_steps),',
            '        "random_start_pct": 0.5,',
            '        "commission": CONFIG.get("commission", 0.003),',
            '        "initial_cash": CONFIG.get("initial_cash", 10000),',
            '    }',
            '',
            '    ray.init(num_cpus=max(2, CONFIG.get("num_rollout_workers", 2) + 2),',
            '             ignore_reinit_error=True, log_to_driver=False)',
            '    register_env("TradingEnv", create_env)',
            '',
            '    # Connect to dashboard',
            '    bridge = None',
            '    try:',
            '        from tensortrade.api.training_bridge import TrainingBridge',
            '        bridge = TrainingBridge()',
            '        bridge.start()',
            f'        bridge.send({{"type": "experiment_start", "experiment_id": EXPERIMENT_ID, "name": "{name}"}})',
            '    except Exception:',
            '        pass',
            '',
            '    ppo_config = (',
            '        PPOConfig()',
            '        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)',
            '        .environment(env="TradingEnv", env_config=env_config)',
            '        .framework("torch")',
            '        .env_runners(num_env_runners=int(CONFIG.get("num_rollout_workers", 2)))',
            '        .callbacks(Callbacks)',
            '        .training(',
            '            lr=CONFIG.get("learning_rate", 5e-5),',
            '            gamma=CONFIG.get("gamma", 0.99),',
            '            lambda_=CONFIG.get("lambda_", 0.95),',
            '            clip_param=CONFIG.get("clip_param", 0.2),',
            '            entropy_coeff=CONFIG.get("entropy_coeff", 0.01),',
            '            train_batch_size=int(CONFIG.get("train_batch_size", 4000)),',
            '            minibatch_size=int(CONFIG.get("sgd_minibatch_size", 128)),',
            '            num_epochs=int(CONFIG.get("num_sgd_iter", 10)),',
            '            vf_clip_param=100.0,',
            '            model=CONFIG.get("model", {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}),',
            '        )',
            '        .resources(num_gpus=0)',
            '    )',
            '',
            '    algo = ppo_config.build()',
            '    num_iterations = int(CONFIG.get("num_iterations", 50))',
            '    start_time = time.time()',
            '',
            '    for i in range(num_iterations):',
            '        result = algo.train()',
            '',
            '        import math',
            '        def safe_float(v, default=0.0):',
            '            if v is None:',
            '                return default',
            '            try:',
            '                f = float(v)',
            '                return default if math.isnan(f) else f',
            '            except (TypeError, ValueError):',
            '                return default',
            '',
            '        # Handle both old and new RLlib API result structures',
            '        er = result.get("env_runners", {})',
            '        custom = er.get("custom_metrics", {}) or result.get("custom_metrics", {})',
            '        ep_reward = safe_float(er.get("episode_return_mean")) or safe_float(er.get("episode_reward_mean")) or safe_float(result.get("episode_reward_mean"))',
            '        metrics = {',
            '            "episode_return_mean": ep_reward,',
            '            "pnl_mean": safe_float(custom.get("pnl_mean")),',
            '            "pnl_pct_mean": safe_float(custom.get("pnl_pct_mean")),',
            '            "net_worth_mean": safe_float(custom.get("final_net_worth_mean")),',
            '            "trade_count_mean": safe_float(custom.get("trade_count_mean")),',
            '            "hold_count_mean": safe_float(custom.get("hold_count_mean")),',
            '            "buy_count_mean": safe_float(custom.get("buy_count_mean")),',
            '            "sell_count_mean": safe_float(custom.get("sell_count_mean")),',
            '        }',
            '',
            '        store.log_iteration(EXPERIMENT_ID, i + 1, metrics)',
            '',
            '        elapsed = time.time() - start_time',
            '        eta = (elapsed / (i + 1)) * (num_iterations - i - 1) if i > 0 else None',
            '',
            '        if bridge:',
            '            try:',
            '                bridge.send({',
            '                    "type": "training_update",',
            '                    "iteration": i + 1,',
            '                    **metrics,',
            '                })',
            '                bridge.send({',
            '                    "type": "training_progress",',
            '                    "experiment_id": EXPERIMENT_ID,',
            '                    "iteration": i + 1,',
            '                    "total_iterations": num_iterations,',
            '                    "elapsed_seconds": elapsed,',
            '                    "eta_seconds": eta,',
            '                })',
            '                bridge.send({',
            '                    "type": "episode_metrics",',
            '                    "episode": i + 1,',
            '                    "reward_total": metrics["episode_return_mean"],',
            '                    "pnl": metrics["pnl_mean"],',
            '                    "pnl_pct": metrics["pnl_pct_mean"],',
            '                    "net_worth": metrics["net_worth_mean"],',
            '                    "trade_count": metrics["trade_count_mean"],',
            '                    "hold_count": metrics["hold_count_mean"],',
            '                    "buy_count": metrics.get("buy_count_mean", 0),',
            '                    "sell_count": metrics.get("sell_count_mean", 0),',
            '                    "action_distribution": {',
            '                        "buy": metrics.get("buy_count_mean", 0),',
            '                        "sell": metrics.get("sell_count_mean", 0),',
            '                        "hold": metrics["hold_count_mean"],',
            '                    },',
            '                })',
            '            except Exception:',
            '                pass',
            '',
            '    # Finalize',
            '    final_metrics = {',
            '        "pnl": metrics.get("pnl_mean", 0),',
            '        "pnl_mean": metrics.get("pnl_mean", 0),',
            '        "pnl_pct": metrics.get("pnl_pct_mean", 0),',
            '        "net_worth": metrics.get("net_worth_mean", 0),',
            '        "episode_return_mean": metrics.get("episode_return_mean", 0),',
            '        "total_iterations": num_iterations,',
            '    }',
            '    store.complete_experiment(EXPERIMENT_ID, "completed", final_metrics)',
            '',
            '    if bridge:',
            '        bridge.send({"type": "experiment_end", "experiment_id": EXPERIMENT_ID, "status": "completed"})',
            '        bridge.stop()',
            '',
            '    algo.stop()',
            '    ray.shutdown()',
            '',
            '    # Cleanup',
            '    if os.path.exists(csv_path):',
            '        os.remove(csv_path)',
            '',
            '',
            'if __name__ == "__main__":',
            '    main()',
        ]

        with open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        return script_path
