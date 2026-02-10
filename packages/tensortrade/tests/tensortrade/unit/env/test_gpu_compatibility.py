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

from tensortrade.env.generic import TradingEnv
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensortrade.env.default as default


class TestGPUCompatibility:
    """Test suite for GPU compatibility (Issue #382)"""

    def test_environment_returns_numpy_arrays(self):
        """Test that environment returns numpy arrays for GPU compatibility"""
        # Create simple environment
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source([100, 101, 102, 103, 104], dtype="float").rename("USD-BTC")
        )
        
        portfolio = Portfolio(USD, [
            Wallet(exchange, 1000 * USD),
            Wallet(exchange, 10 * BTC)
        ])
        
        feed = DataFeed([
            Stream.source([100, 101, 102, 103, 104], dtype="float").rename("price")
        ])
        
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=1
        )
        
        # Test reset returns numpy array
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray), f"Expected numpy array, got {type(obs)}"
        
        # Test step returns numpy array
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray), f"Expected numpy array, got {type(obs)}"
        
    def test_ensure_numpy_with_numpy_input(self):
        """Test _ensure_numpy method with numpy array input"""
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source([100], dtype="float").rename("USD-BTC")
        )
        
        portfolio = Portfolio(USD, [Wallet(exchange, 1000 * USD)])
        feed = DataFeed([Stream.source([100], dtype="float").rename("price")])
        
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=1
        )
        
        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = env._ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
        
    def test_ensure_numpy_with_list_input(self):
        """Test _ensure_numpy method with list input"""
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source([100], dtype="float").rename("USD-BTC")
        )
        
        portfolio = Portfolio(USD, [Wallet(exchange, 1000 * USD)])
        feed = DataFeed([Stream.source([100], dtype="float").rename("price")])
        
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=1
        )
        
        # Test with list
        lst = [1, 2, 3]
        result = env._ensure_numpy(lst)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array(lst))
        
    def test_device_parameter(self):
        """Test that device parameter can be set"""
        exchange = Exchange("simulated", service=execute_order)(
            Stream.source([100], dtype="float").rename("USD-BTC")
        )
        
        portfolio = Portfolio(USD, [Wallet(exchange, 1000 * USD)])
        feed = DataFeed([Stream.source([100], dtype="float").rename("price")])
        
        env = default.create(
            portfolio=portfolio,
            action_scheme="simple",
            reward_scheme="simple",
            feed=feed,
            window_size=1,
            device="cpu"
        )
        
        # Verify device is set
        assert env.device == "cpu"

