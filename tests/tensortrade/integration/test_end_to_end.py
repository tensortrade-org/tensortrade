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

import pytest
import numpy as np

from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensortrade.env.default as default


class TestEndToEnd:
    """Integration tests for complete trading workflows"""

    def test_complete_training_workflow(self):
        """Test complete training workflow from environment creation to episode completion"""
        # Create price data
        prices = np.random.uniform(90, 110, 100).tolist()
        
        # Create exchange
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source(prices, dtype="float").rename("USD-BTC")
        )
        
        # Create portfolio
        portfolio = Portfolio(USD, [
            Wallet(exchange, 10000 * USD),
            Wallet(exchange, 0 * BTC)
        ])
        
        # Create feed
        feed = DataFeed([
            Stream.source(prices, dtype="float").rename("price"),
            Stream.source(np.random.uniform(1000, 2000, 100).tolist(), dtype="float").rename("volume")
        ])
        
        # Create environment
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=5,
            max_allowed_loss=0.5
        )
        
        # Run episode
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        
        total_reward = 0
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(info, dict)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        print(f"Episode completed in {steps} steps with total reward: {total_reward}")
        
    def test_multiple_instruments(self):
        """Test environment with multiple trading instruments"""
        # Create price data
        btc_prices = np.random.uniform(90, 110, 50).tolist()
        eth_prices = np.random.uniform(1800, 2200, 50).tolist()
        
        # Create exchange
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source(btc_prices, dtype="float").rename("USD-BTC"),
            Stream.source(eth_prices, dtype="float").rename("USD-ETH")
        )
        
        # Create portfolio
        portfolio = Portfolio(USD, [
            Wallet(exchange, 10000 * USD),
            Wallet(exchange, 0 * BTC),
            Wallet(exchange, 0 * ETH)
        ])
        
        # Create feed
        feed = DataFeed([
            Stream.source(btc_prices, dtype="float").rename("btc_price"),
            Stream.source(eth_prices, dtype="float").rename("eth_price")
        ])
        
        # Create environment
        env = default.create(
            portfolio=portfolio,
            action_scheme="managed-risk",
            reward_scheme="risk-adjusted",
            feed=feed,
            window_size=5
        )
        
        # Test environment
        obs, info = env.reset()
        assert obs is not None
        
        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
    def test_portfolio_operations(self):
        """Test portfolio balance tracking and order execution"""
        # Create simple environment
        prices = [100, 101, 102, 103, 104, 105]
        
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source(prices, dtype="float").rename("USD-BTC")
        )
        
        initial_usd = 10000
        portfolio = Portfolio(USD, [
            Wallet(exchange, initial_usd * USD),
            Wallet(exchange, 0 * BTC)
        ])
        
        feed = DataFeed([
            Stream.source(prices, dtype="float").rename("price")
        ])
        
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=1
        )
        
        # Reset environment
        obs, info = env.reset()
        
        # Get initial net worth
        initial_net_worth = portfolio.net_worth
        
        # Execute some trades
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Verify portfolio state is tracked
        assert portfolio.net_worth > 0
        print(f"Initial net worth: {initial_net_worth}, Final net worth: {portfolio.net_worth}")
        
    def test_data_feed_processing(self):
        """Test data feed processing with multiple features"""
        # Create multi-feature data
        n_samples = 50
        prices = np.random.uniform(90, 110, n_samples).tolist()
        volumes = np.random.uniform(1000, 2000, n_samples).tolist()
        highs = [p * 1.02 for p in prices]
        lows = [p * 0.98 for p in prices]
        
        # Create exchange
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source(prices, dtype="float").rename("USD-BTC")
        )
        
        # Create portfolio
        portfolio = Portfolio(USD, [
            Wallet(exchange, 10000 * USD),
            Wallet(exchange, 0 * BTC)
        ])
        
        # Create feed with multiple features
        feed = DataFeed([
            Stream.source(prices, dtype="float").rename("close"),
            Stream.source(highs, dtype="float").rename("high"),
            Stream.source(lows, dtype="float").rename("low"),
            Stream.source(volumes, dtype="float").rename("volume")
        ])
        
        # Create environment
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=10
        )
        
        # Test environment
        obs, info = env.reset()
        assert obs.shape[0] == 10  # window_size
        assert obs.shape[1] >= 4  # At least 4 features
        
        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape[0] == 10  # window_size maintained
            if terminated or truncated:
                break

