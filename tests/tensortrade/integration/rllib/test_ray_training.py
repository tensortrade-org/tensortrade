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

"""Tests for Ray RLlib training integration."""

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
class TestPPOTraining:
    """Tests for PPO algorithm training."""

    def test_ppo_single_iteration(self, minimal_env_config: dict, ray_session):
        """PPO can complete a single training iteration."""
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
            .env_runners(num_env_runners=0)  # Use main process only
            .training(
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                train_batch_size=200,
                minibatch_size=32,
                num_sgd_iter=1,
            )
            .resources(num_gpus=0)
        )

        algo = config.build()
        result = algo.train()
        algo.stop()

        # Verify training produced metrics
        assert result is not None
        assert "env_runners" in result or "num_env_steps_sampled_lifetime" in result

    def test_ppo_with_lstm_model(self, minimal_env_config: dict, ray_session):
        """PPO with LSTM model can initialize and run."""
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
                gamma=0.99,
                lambda_=0.95,
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

        algo = config.build()
        result = algo.train()
        algo.stop()

        assert result is not None

    def test_ppo_with_attention_model(self, minimal_env_config: dict, ray_session):
        """PPO with AttentionNet model can initialize and run."""
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
                gamma=0.99,
                lambda_=0.95,
                train_batch_size=200,
                minibatch_size=32,
                num_sgd_iter=1,
                model={
                    "use_attention": True,
                    "max_seq_len": 5,
                    "attention_num_transformer_units": 1,
                    "attention_dim": 16,
                    "attention_memory_inference": 5,
                    "attention_memory_training": 5,
                    "attention_num_heads": 1,
                    "attention_head_dim": 16,
                    "attention_position_wise_mlp_dim": 16,
                },
            )
            .resources(num_gpus=0)
        )

        algo = config.build()
        result = algo.train()
        algo.stop()

        assert result is not None


@pytest.mark.rllib
class TestTunerConfig:
    """Tests for Ray Tune configuration validation."""

    def test_tuner_config_validation(self, minimal_env_config: dict, ray_session):
        """Tuner configuration is valid and can be created."""
        from ray import tune
        from ray.tune.registry import register_env

        register_env("TradingEnv", create_env)

        # This tests that the config structure is valid without running training
        param_space = {
            "env": "TradingEnv",
            "env_config": minimal_env_config,
            "framework": "torch",
            "num_env_runners": 0,
            "num_gpus": 0,
            "lr": tune.loguniform(1e-5, 1e-2),
            "gamma": tune.uniform(0.8, 0.99),
            "lambda_": tune.uniform(0.1, 0.8),
            "model": {
                "use_lstm": True,
                "lstm_cell_size": 64,
            },
        }

        from ray.train import CheckpointConfig

        tune_config = tune.TuneConfig(
            num_samples=1,
            metric="env_runners/episode_reward_mean",
            mode="max",
        )

        checkpoint_config = CheckpointConfig(
            num_to_keep=1,
        )

        # Verify configs can be created without error
        assert param_space is not None
        assert tune_config is not None
        assert checkpoint_config is not None
