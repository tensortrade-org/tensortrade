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

"""Tests for TensorTrade environment creation and Gymnasium compliance."""

import gymnasium as gym
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


def create_env(config: dict) -> gym.Env:
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


class TestEnvCreation:
    """Tests for environment creation."""

    def test_create_env_returns_valid_gymnasium_env(self, minimal_env_config: dict):
        """Environment creation returns a valid Gymnasium environment."""
        env = create_env(minimal_env_config)
        assert isinstance(env, gym.Env)
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

    def test_env_observation_space_shape(self, minimal_env_config: dict):
        """Observation space has correct shape based on window_size."""
        env = create_env(minimal_env_config)
        obs, _ = env.reset()

        window_size = minimal_env_config["window_size"]
        assert obs.shape[0] == window_size
        assert len(obs.shape) == 2  # (window_size, n_features)

    def test_env_action_space_discrete(self, minimal_env_config: dict):
        """Action space is discrete for BSH action scheme."""
        env = create_env(minimal_env_config)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 2  # BSH toggles between buy (hold asset) and sell (hold cash)

    def test_env_reset_returns_tuple(self, minimal_env_config: dict):
        """Environment reset returns (observation, info) tuple per Gymnasium API."""
        env = create_env(minimal_env_config)
        result = env.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2

        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_env_step_returns_5_tuple(self, minimal_env_config: dict):
        """Environment step returns 5-tuple per Gymnasium API."""
        env = create_env(minimal_env_config)
        env.reset()

        action = env.action_space.sample()
        result = env.step(action)

        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)


@pytest.mark.rllib
class TestRayEnvRegistration:
    """Tests for Ray environment registration."""

    def test_env_registration_with_ray(self, minimal_env_config: dict, ray_session):
        """Environment can be registered with Ray."""
        from ray.tune.registry import register_env

        register_env("TradingEnv", create_env)

        # Verify registration works by checking Ray can create the env
        from ray.rllib.env.env_context import EnvContext
        env_context = EnvContext(
            env_config=minimal_env_config,
            worker_index=0,
            num_workers=1,
        )

        env = create_env(env_context)
        assert isinstance(env, gym.Env)

    def test_registered_env_reset_and_step(self, minimal_env_config: dict, ray_session):
        """Registered environment can reset and step correctly."""
        from ray.tune.registry import register_env

        register_env("TradingEnv", create_env)

        env = create_env(minimal_env_config)
        obs, info = env.reset()
        assert obs is not None

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert obs is not None
