"""
Inference runner for replay/playback of a trained or random agent.

Loads an experiment config, creates a TradingEnv, runs a step loop
with pacing, and broadcasts step/trade messages via the ConnectionManager.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

import tensortrade.env.default as default
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.env.default import actions as tt_actions
from tensortrade.env.default import rewards as tt_rewards
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import BTC, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.training.feature_engine import FeatureEngine

if TYPE_CHECKING:
    from tensortrade.api.server import ConnectionManager
    from tensortrade.training.dataset_store import DatasetStore
    from tensortrade.training.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


class InferenceRunner:
    """Runs a single trained-policy inference episode and streams results."""

    def __init__(
        self,
        store: ExperimentStore,
        manager: ConnectionManager,
        dataset_store: DatasetStore,
    ) -> None:
        self.store = store
        self.manager = manager
        self.dataset_store = dataset_store

    async def run_episode(
        self,
        experiment_id: str,
        dataset_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        test_only: bool = False,
    ) -> None:
        """Run one episode and broadcast step/trade updates.

        Args:
            experiment_id: The experiment whose config to use.
            dataset_id: Optional override dataset ID. If provided, load
                this dataset's config instead of the experiment's embedded one.
            start_date: Optional start date (YYYY-MM-DD) to filter data.
            end_date: Optional end date (YYYY-MM-DD) to filter data.
            test_only: If True, restrict data to the test split (last 15%).
        """
        policy_algo = None
        ray_started_here = False
        try:
            exp = self.store.get_experiment(experiment_id)
            if not exp:
                await self._send_error(experiment_id, "Experiment not found")
                return

            config = exp.config

            policy_algo, ray_started_here = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_trained_algo,
                exp,
            )

            # Resolve dataset config: override or from experiment
            dataset_name = self._resolve_dataset_name(config, dataset_id)

            env, ohlcv, asset_wallet = await asyncio.get_event_loop().run_in_executor(
                None,
                self._create_env,
                config,
                dataset_id,
                start_date,
                end_date,
                test_only,
            )

            # The env's window_size consumes initial rows, so the first
            # step corresponds to row window_size in the OHLCV data.
            window_size = int(getattr(env, "window_size", 10))
            estimated_total = max(len(ohlcv) - window_size - 1, 0)

            await self.manager.broadcast_to_dashboards(
                {
                    "type": "inference_status",
                    "state": "running",
                    "experiment_id": experiment_id,
                    "total_steps": estimated_total,
                    "current_step": 0,
                    "source": "inference",
                    "dataset_name": dataset_name,
                }
            )

            obs, _ = env.reset()
            done = truncated = False
            step = 0

            initial_net_worth = float(env.portfolio.net_worth)
            peak_net_worth = initial_net_worth
            max_drawdown_pct = 0.0
            buy_count = 0
            sell_count = 0
            hold_count = 0
            total_trades = 0

            # Track broker trades by count to detect new ones each step
            broker = env.action_scheme.broker
            prev_trade_count = sum(len(tl) for tl in broker.trades.values())

            while not done and not truncated:
                action = self._coerce_action(policy_algo.compute_single_action(obs, explore=False))

                obs, reward, done, truncated, info = env.step(action)
                step += 1
                net_worth = float(env.portfolio.net_worth)
                peak_net_worth = max(peak_net_worth, net_worth)
                if peak_net_worth > 0:
                    drawdown_pct = ((peak_net_worth - net_worth) / peak_net_worth) * 100
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

                # Look up OHLCV from the original data by step index
                data_idx = window_size + step - 1
                if data_idx >= len(ohlcv):
                    break  # No more real price data — stop cleanly
                row = ohlcv.iloc[data_idx]
                o = float(row.get("open", 0))
                h = float(row.get("high", 0))
                lo = float(row.get("low", 0))
                c = float(row.get("close", 0))
                v = float(row.get("volume", 0))
                ts = int(pd.Timestamp(row["date"]).timestamp()) if "date" in row.index else step

                # Broadcast executed trades from the OMS broker
                all_trades = [t for tl in broker.trades.values() for t in tl]
                new_trades = all_trades[prev_trade_count:]
                for trade in new_trades:
                    total_trades += 1
                    side = str(trade.side)  # TradeSide.BUY/SELL → "buy"/"sell"
                    if side == "buy":
                        buy_count += 1
                    else:
                        sell_count += 1
                    commission_val = float(trade.commission.size) if hasattr(trade.commission, "size") else 0.0
                    await self.manager.broadcast_to_dashboards(
                        {
                            "type": "trade",
                            "step": step,
                            "timestamp": ts,
                            "side": side,
                            "price": float(trade.price),
                            "size": round(float(trade.size), 8),
                            "commission": round(commission_val, 8),
                            "source": "inference",
                        }
                    )
                if not new_trades:
                    hold_count += 1
                prev_trade_count = len(all_trades)
                await self.manager.broadcast_to_dashboards(
                    {
                        "type": "step_update",
                        "step": step,
                        "timestamp": ts,
                        "price": c,
                        "open": o,
                        "high": h,
                        "low": lo,
                        "close": c,
                        "volume": v,
                        "net_worth": net_worth,
                        "action": action,
                        "reward": float(reward),
                        "source": "inference",
                    }
                )

                # Pacing: yield control and add small delay
                await asyncio.sleep(0.05)

            # Episode complete - send summary
            final_net_worth = float(env.portfolio.net_worth)
            pnl = final_net_worth - initial_net_worth
            pnl_pct = (pnl / initial_net_worth) * 100 if initial_net_worth > 0 else 0.0
            total_actions = buy_count + sell_count + hold_count
            hold_ratio = (hold_count / total_actions) if total_actions > 0 else 0.0
            trade_ratio = (total_trades / total_actions) if total_actions > 0 else 0.0
            pnl_per_trade = (pnl / total_trades) if total_trades > 0 else 0.0

            await self.manager.broadcast_to_dashboards(
                {
                    "type": "inference_status",
                    "state": "completed",
                    "experiment_id": experiment_id,
                    "total_steps": step,
                    "current_step": step,
                    "source": "inference",
                    "dataset_name": dataset_name,
                    "episode_summary": {
                        "total_steps": step,
                        "final_net_worth": final_net_worth,
                        "initial_net_worth": initial_net_worth,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "total_trades": total_trades,
                        "buy_count": buy_count,
                        "sell_count": sell_count,
                        "hold_count": hold_count,
                        "hold_ratio": hold_ratio,
                        "trade_ratio": trade_ratio,
                        "pnl_per_trade": pnl_per_trade,
                        "max_drawdown_pct": max_drawdown_pct,
                    },
                }
            )

        except Exception as e:
            logger.exception("Inference run failed")
            await self._send_error(experiment_id, str(e))
        finally:
            if policy_algo is not None and hasattr(policy_algo, "stop"):
                try:
                    policy_algo.stop()
                except Exception:
                    logger.debug("Failed to stop inference policy cleanly", exc_info=True)
            from tensortrade.ray_manager import ray_manager

            ray_manager.release("inference")

    async def _send_error(self, experiment_id: str, message: str) -> None:
        await self.manager.broadcast_to_dashboards(
            {
                "type": "inference_status",
                "state": "error",
                "experiment_id": experiment_id,
                "total_steps": 0,
                "current_step": 0,
                "source": "inference",
                "error": message,
            }
        )

    def _resolve_dataset_name(
        self,
        config: dict,
        dataset_id: str | None,
    ) -> str:
        """Get a human-readable dataset name for display."""
        if dataset_id:
            ds = self.dataset_store.get_config(dataset_id)
            if ds:
                return ds.name
            return f"Dataset {dataset_id[:8]}"
        return str(config.get("dataset_name", "Unknown"))

    @staticmethod
    def _coerce_action(action_value: object) -> int:
        """Normalize RLlib action outputs to a plain integer action."""
        value = action_value
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, list):
            value = value[0] if value else 0
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        return int(value)

    @staticmethod
    def _get_checkpoint_path(experiment: object) -> str | None:
        config = getattr(experiment, "config", {}) or {}
        final_metrics = getattr(experiment, "final_metrics", {}) or {}
        training_config = config.get("training_config", {}) if isinstance(config, dict) else {}

        candidate_paths = [
            final_metrics.get("checkpoint_path"),
            config.get("checkpoint_path") if isinstance(config, dict) else None,
            training_config.get("checkpoint_path") if isinstance(training_config, dict) else None,
        ]
        for path in candidate_paths:
            if isinstance(path, str) and path.strip():
                return path
        return None

    def _load_trained_algo(self, experiment: object):
        checkpoint_path = self._get_checkpoint_path(experiment)
        if not checkpoint_path:
            raise ValueError(
                "No checkpoint found for this experiment. Re-run training so a checkpoint can be saved.",
            )
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        from ray.rllib.policy.policy import Policy

        from tensortrade.ray_manager import ray_manager

        ray_manager.acquire("inference")

        # Load just the policy weights — no env runners or env registration needed.
        # Policy.from_checkpoint returns a dict of {policy_id: Policy}.
        policies = Policy.from_checkpoint(checkpoint_path)
        policy = policies.get("default_policy")
        if policy is None:
            raise ValueError(f"No default_policy found in checkpoint: {checkpoint_path}")
        return policy, False  # ray_started_here unused now

    def _create_env(
        self,
        config: dict,
        dataset_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        test_only: bool = False,
    ):
        """Create a TradingEnv from experiment config using proper dataset loading.

        Returns (env, ohlcv_data) where ohlcv_data is the OHLCV DataFrame
        used to build the env, so the caller can look up prices by step.

        Args:
            config: Experiment config dict.
            dataset_id: Optional override dataset ID.
            start_date: Optional start date (YYYY-MM-DD) to filter data.
            end_date: Optional end date (YYYY-MM-DD) to filter data.
        """
        # Resolve dataset settings: override or experiment-embedded
        # When overriding dataset, use the override's source data but always
        # use the experiment's feature spec (the model expects those features).
        trained_features = config.get("features", [])
        if dataset_id:
            ds = self.dataset_store.get_config(dataset_id)
            if ds is None:
                raise ValueError(f"Dataset not found: {dataset_id}")
            source_type = ds.source_type
            source_config = ds.source_config
            features_spec = trained_features if trained_features else ds.features
        else:
            source_type = config.get("source_type", "crypto_download")
            source_config = config.get("source_config", {})
            features_spec = trained_features

        # Load data based on source type
        data = self._load_data(source_type, source_config)

        # Compute features using FeatureEngine
        engine = FeatureEngine()
        if features_spec:
            data = engine.compute(data, features_spec)
            feature_cols = engine.get_feature_columns(features_spec)
        else:
            # Fallback: use all non-OHLCV columns
            ohlcv = {"date", "open", "high", "low", "close", "volume"}
            feature_cols = [c for c in data.columns if c not in ohlcv]

        # Filter data: test_only uses the test split, date range overrides, else last 1000
        if test_only:
            split_config = config.get("split_config", {})
            test_pct = split_config.get("test_pct", 0.15)
            n = len(data)
            test_start = int(n * (1 - test_pct))
            data = data.iloc[test_start:].reset_index(drop=True)
        elif start_date or end_date:
            if "date" in data.columns:
                if start_date:
                    data = data[data["date"] >= pd.Timestamp(start_date)]
                if end_date:
                    data = data[data["date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
            data = data.reset_index(drop=True)
        else:
            data = data.tail(1000).reset_index(drop=True)

        # Keep OHLCV + date for step broadcasts
        ohlcv_cols = ["date", "open", "high", "low", "close", "volume"]
        ohlcv_data = data[[c for c in ohlcv_cols if c in data.columns]].copy()

        # Build environment
        training_config = config.get("training_config", config)
        commission = training_config.get("commission", config.get("commission", 0.0005))
        initial_cash = training_config.get("initial_cash", config.get("initial_cash", 10000))

        price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
        exchange = Exchange(
            "exchange",
            service=execute_order,
            options=ExchangeOptions(commission=commission),
        )(price)

        cash = Wallet(exchange, initial_cash * USD)
        asset = Wallet(exchange, 0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        features = [Stream.source(list(data[c]), dtype="float").rename(c) for c in feature_cols if c in data.columns]
        feed = DataFeed(features)
        feed.compile()

        # Dispatch reward scheme from config
        reward_name = training_config.get("reward_scheme", config.get("reward_scheme", "PBR"))
        reward_params = training_config.get("reward_params", config.get("reward_params", {}))
        anti_churn_defaults = {
            "trade_penalty_multiplier": 1.1,
            "churn_penalty_multiplier": 1.0,
            "churn_window": 6,
            "reward_clip": 200.0,
        }
        if reward_name == "PBR":
            reward_scheme = tt_rewards.PBR(
                price=price,
                commission=commission,
                **{**anti_churn_defaults, **reward_params},
            )
        elif reward_name == "AdvancedPBR":
            reward_scheme = tt_rewards.AdvancedPBR(
                price=price,
                commission=commission,
                **{**anti_churn_defaults, **reward_params},
            )
        elif reward_name == "FractionalPBR":
            reward_scheme = tt_rewards.FractionalPBR(price=price, commission=commission)
        elif reward_name == "RiskAdjustedReturns":
            reward_scheme = tt_rewards.RiskAdjustedReturns(**reward_params)
        elif reward_name == "MaxDrawdownPenalty":
            reward_scheme = tt_rewards.MaxDrawdownPenalty(**reward_params)
        elif reward_name == "AdaptiveProfitSeeker":
            reward_scheme = tt_rewards.AdaptiveProfitSeeker(price=price, commission=commission, **reward_params)
        else:
            reward_scheme = tt_rewards.SimpleProfit(**reward_params)

        # Dispatch action scheme from config
        action_name = training_config.get("action_scheme", config.get("action_scheme", "BSH"))
        action_cls = getattr(tt_actions, action_name, tt_actions.BSH)
        action_scheme = action_cls(cash=cash, asset=asset)
        if hasattr(reward_scheme, "on_action"):
            action_scheme.attach(reward_scheme)

        env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            window_size=training_config.get("window_size", config.get("window_size", 10)),
            max_allowed_loss=training_config.get("max_allowed_loss", config.get("max_allowed_loss", 0.4)),
        )
        env.portfolio = portfolio
        return env, ohlcv_data, asset

    @staticmethod
    def _load_data(source_type: str, source_config: dict) -> pd.DataFrame:
        """Load data based on source type and config, matching launcher.py pattern."""
        if source_type == "crypto_download":
            cdd = CryptoDataDownload()
            data = cdd.fetch(
                source_config.get("exchange", "Bitfinex"),
                source_config.get("base", "BTC"),
                source_config.get("quote", "USD"),
                source_config.get("timeframe", "1h"),
            )
            data = data[["date", "open", "high", "low", "close", "volume"]]
            data["date"] = pd.to_datetime(data["date"])
            data.sort_values("date", inplace=True)
            data.reset_index(drop=True, inplace=True)
        elif source_type == "alpaca_crypto":
            from tensortrade.data.alpaca_crypto import AlpacaCryptoData

            alpaca = AlpacaCryptoData()
            data = alpaca.fetch(
                symbol=source_config.get("symbol", "BTC/USD"),
                timeframe=source_config.get("timeframe", "1h"),
                start_date=source_config.get("start_date", ""),
                end_date=source_config.get("end_date", ""),
            )
        elif source_type == "synthetic":
            from tensortrade.stochastic.processes.gbm import gbm

            data = gbm(
                base_price=source_config.get("base_price", 50000),
                base_volume=source_config.get("base_volume", 1000),
                start_date="2020-01-01",
                times_to_generate=source_config.get("num_candles", 5000),
                time_frame=source_config.get("timeframe", "1h"),
            )
            data.index.name = "date"
            data = data.reset_index()
            data["date"] = pd.to_datetime(data["date"])
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        return data
