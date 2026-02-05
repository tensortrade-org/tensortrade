# Copyright 2025 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Tests for RLlib checkpoint save/restore functionality."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default


def create_env(config: dict):
    """Create a TensorTrade environment from config."""
    data = pd.read_csv(
        filepath_or_buffer=config["csv_filename"],
        parse_dates=['date']
    ).bfill().ffill()

    commission = config.get("commission", 0.001)
    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    exchange_options = ExchangeOptions(commission=commission)
    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=exchange_options
    )(price)

    initial_cash = config.get("initial_cash", 10000)
    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [
        Stream.source(list(data[c]), dtype="float").rename(c)
        for c in ['open', 'high', 'low', 'close', 'volume']
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
        max_allowed_loss=config.get("max_allowed_loss", 0.5)
    )

    return env


@pytest.mark.rllib
@pytest.mark.slow
class TestCheckpoint:
    """Tests for checkpoint save and restore."""

    def test_checkpoint_save_creates_files(
        self, minimal_env_config: dict, ray_session, tmp_path: Path
    ):
        """Saving a checkpoint creates checkpoint files."""
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPOConfig

        register_env("TradingEnv", create_env)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="TradingEnv", env_config=minimal_env_config)
            .framework("torch")
            .env_runners(num_env_runners=0)
            .training(
                lr=3e-4,
                train_batch_size=200,
                minibatch_size=32,
                num_sgd_iter=1,
            )
            .resources(num_gpus=0)
        )

        algo = config.build()
        algo.train()

        # Save checkpoint
        checkpoint_dir = str(tmp_path / "checkpoint")
        checkpoint_result = algo.save(checkpoint_dir)
        algo.stop()

        # Verify checkpoint was created
        assert checkpoint_result is not None
        assert checkpoint_result.checkpoint is not None

    def test_checkpoint_restore_loads_algorithm(
        self, minimal_env_config: dict, ray_session, tmp_path: Path
    ):
        """Restoring from checkpoint loads the algorithm correctly."""
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPO, PPOConfig

        register_env("TradingEnv", create_env)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="TradingEnv", env_config=minimal_env_config)
            .framework("torch")
            .env_runners(num_env_runners=0)
            .training(
                lr=3e-4,
                train_batch_size=200,
                minibatch_size=32,
                num_sgd_iter=1,
            )
            .resources(num_gpus=0)
        )

        # Create and train original algorithm
        algo = config.build()
        algo.train()
        checkpoint_dir = str(tmp_path / "checkpoint")
        checkpoint_result = algo.save(checkpoint_dir)
        algo.stop()

        # Restore from checkpoint
        restored_algo = PPO.from_checkpoint(checkpoint_result.checkpoint)

        # Verify restored algorithm works
        assert restored_algo is not None
        assert restored_algo.get_policy() is not None

        restored_algo.stop()

    def test_restored_algo_computes_actions(
        self, minimal_env_config: dict, ray_session, tmp_path: Path
    ):
        """Restored algorithm can compute actions."""
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPO, PPOConfig

        register_env("TradingEnv", create_env)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="TradingEnv", env_config=minimal_env_config)
            .framework("torch")
            .env_runners(num_env_runners=0)
            .training(
                lr=3e-4,
                train_batch_size=200,
                minibatch_size=32,
                num_sgd_iter=1,
            )
            .resources(num_gpus=0)
        )

        # Create, train, and save
        algo = config.build()
        algo.train()
        checkpoint_dir = str(tmp_path / "checkpoint")
        checkpoint_result = algo.save(checkpoint_dir)
        algo.stop()

        # Restore
        restored_algo = PPO.from_checkpoint(checkpoint_result.checkpoint)

        # Create environment and test action computation
        env = create_env(minimal_env_config)
        obs, _ = env.reset()

        # Compute action
        action = restored_algo.compute_single_action(obs)

        # Verify action is valid
        assert action is not None
        assert action in [0, 1]  # BSH has 2 actions (toggle between holding cash vs asset)

        restored_algo.stop()


@pytest.mark.rllib
@pytest.mark.slow
class TestLSTMCheckpoint:
    """Tests for LSTM model checkpoint save/restore."""

    def test_lstm_checkpoint_restore(
        self, minimal_env_config: dict, ray_session, tmp_path: Path
    ):
        """LSTM model checkpoint can be saved and restored."""
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPO, PPOConfig

        register_env("TradingEnv", create_env)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="TradingEnv", env_config=minimal_env_config)
            .framework("torch")
            .env_runners(num_env_runners=0)
            .training(
                lr=3e-4,
                train_batch_size=200,
                minibatch_size=32,
                num_sgd_iter=1,
                model={
                    "use_lstm": True,
                    "lstm_cell_size": 64,
                },
            )
            .resources(num_gpus=0)
        )

        # Create, train, save
        algo = config.build()
        algo.train()
        checkpoint_dir = str(tmp_path / "lstm_checkpoint")
        checkpoint_result = algo.save(checkpoint_dir)
        algo.stop()

        # Restore
        restored_algo = PPO.from_checkpoint(checkpoint_result.checkpoint)

        # Test with LSTM state
        env = create_env(minimal_env_config)
        obs, _ = env.reset()

        lstm_cell_size = 64
        state = [np.zeros(lstm_cell_size), np.zeros(lstm_cell_size)]

        action, state_out, _ = restored_algo.compute_single_action(
            obs, state=state, full_fetch=True
        )

        assert action is not None
        assert state_out is not None
        assert len(state_out) == 2

        restored_algo.stop()
