"""
Inference runner for replay/playback of a trained or random agent.

Loads an experiment config, creates a TradingEnv, runs a step loop
with pacing, and broadcasts step/trade messages via the ConnectionManager.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default

if TYPE_CHECKING:
    from tensortrade.api.server import ConnectionManager
    from tensortrade.training.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


class InferenceRunner:
    """Runs a single inference episode and streams results to dashboards."""

    def __init__(self, store: ExperimentStore, manager: ConnectionManager) -> None:
        self.store = store
        self.manager = manager

    async def run_episode(
        self,
        experiment_id: str,
        use_random_agent: bool = True,
    ) -> None:
        """Run one episode and broadcast step/trade updates."""
        try:
            await self.manager.broadcast_to_dashboards({
                "type": "inference_status",
                "state": "running",
                "experiment_id": experiment_id,
                "total_steps": 0,
                "current_step": 0,
                "source": "inference",
            })

            exp = self.store.get_experiment(experiment_id)
            if not exp:
                await self._send_error(experiment_id, "Experiment not found")
                return

            config = exp.config
            env = await asyncio.get_event_loop().run_in_executor(
                None, self._create_env, config
            )

            obs, _ = env.reset()
            done = truncated = False
            step = 0
            initial_net_worth = float(env.portfolio.net_worth)
            prev_net_worth = initial_net_worth
            buy_count = 0
            sell_count = 0
            hold_count = 0
            total_trades = 0

            while not done and not truncated:
                if use_random_agent:
                    action = int(env.action_space.sample())
                else:
                    action = int(env.action_space.sample())

                obs, reward, done, truncated, info = env.step(action)
                step += 1
                net_worth = float(env.portfolio.net_worth)

                # Track action distribution
                if action == 0:
                    hold_count += 1
                elif action == 1:
                    buy_count += 1
                elif action == 2:
                    sell_count += 1

                # Detect trades by net worth change threshold
                nw_change = abs(net_worth - prev_net_worth)
                if action in (1, 2) and nw_change > 0.01:
                    total_trades += 1
                    side = "buy" if action == 1 else "sell"
                    price = float(obs[3]) if len(obs) > 3 else 0.0
                    await self.manager.broadcast_to_dashboards({
                        "type": "trade",
                        "step": step,
                        "side": side,
                        "price": price,
                        "size": 1.0,
                        "commission": 0.0,
                        "source": "inference",
                    })

                # Build OHLCV from observation if possible
                price = float(obs[3]) if len(obs) > 3 else net_worth
                await self.manager.broadcast_to_dashboards({
                    "type": "step_update",
                    "step": step,
                    "price": price,
                    "open": float(obs[0]) if len(obs) > 0 else price,
                    "high": float(obs[1]) if len(obs) > 1 else price,
                    "low": float(obs[2]) if len(obs) > 2 else price,
                    "close": price,
                    "volume": float(obs[4]) if len(obs) > 4 else 0.0,
                    "net_worth": net_worth,
                    "action": action,
                    "reward": float(reward),
                    "source": "inference",
                })

                prev_net_worth = net_worth

                # Pacing: yield control and add small delay
                await asyncio.sleep(0.05)

            # Episode complete - send summary
            final_net_worth = float(env.portfolio.net_worth)
            pnl = final_net_worth - initial_net_worth
            pnl_pct = (pnl / initial_net_worth) * 100 if initial_net_worth > 0 else 0.0

            await self.manager.broadcast_to_dashboards({
                "type": "inference_status",
                "state": "completed",
                "experiment_id": experiment_id,
                "total_steps": step,
                "current_step": step,
                "source": "inference",
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
                },
            })

        except Exception as e:
            logger.exception("Inference run failed")
            await self._send_error(experiment_id, str(e))

    async def _send_error(self, experiment_id: str, message: str) -> None:
        await self.manager.broadcast_to_dashboards({
            "type": "inference_status",
            "state": "error",
            "experiment_id": experiment_id,
            "total_steps": 0,
            "current_step": 0,
            "source": "inference",
            "error": message,
        })

    @staticmethod
    def _create_env(config: dict):
        """Create a TradingEnv from experiment config."""
        # Fetch fresh data
        cdd = CryptoDataDownload()
        data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
        data = data[["date", "open", "high", "low", "close", "volume"]]
        data["date"] = pd.to_datetime(data["date"])
        data.sort_values("date", inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Add basic features
        for p in [1, 4, 12, 24]:
            data[f"ret_{p}h"] = np.tanh(data["close"].pct_change(p) * 10)

        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data["rsi"] = (100 - (100 / (1 + rs)) - 50) / 50

        data = data.bfill().ffill()

        feature_cols = [
            c for c in data.columns
            if c not in ["date", "open", "high", "low", "close", "volume"]
        ]

        # Use last 1000 candles for inference
        data = data.tail(1000).reset_index(drop=True)

        price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
        commission = config.get("commission", 0.0005)
        exchange = Exchange(
            "exchange",
            service=execute_order,
            options=ExchangeOptions(commission=commission),
        )(price)

        initial_cash = config.get("initial_cash", 10000)
        cash = Wallet(exchange, initial_cash * USD)
        asset = Wallet(exchange, 0 * BTC)
        portfolio = Portfolio(USD, [cash, asset])

        features = [
            Stream.source(list(data[c]), dtype="float").rename(c) for c in feature_cols
        ]
        feed = DataFeed(features)
        feed.compile()

        reward_scheme = PBR(price=price)
        action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

        env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            window_size=config.get("window_size", 10),
            max_allowed_loss=config.get("max_allowed_loss", 0.4),
        )
        env.portfolio = portfolio
        return env
