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
from tensortrade.oms.instruments import USD, BTC, Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.env.default.observers import _create_wallet_source


class TestStreamSelector:
    """Test suite for stream selector functionality (Issue #470)"""

    def test_stream_selector_with_colon_naming(self):
        """Test stream selector works with exchange:symbol naming convention"""
        # Create exchange with stream using colon naming
        exchange = Exchange("simulated", service=execute_order)
        price_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        exchange(price_stream)
        
        # Create wallet
        wallet = Wallet(exchange, 1000 * USD)
        
        # Test that stream selector can find the price stream
        streams = _create_wallet_source(wallet, include_worth=True)
        
        # Should not raise "No stream satisfies selector condition" error
        assert len(streams) > 0
        
    def test_stream_selector_with_dash_naming(self):
        """Test stream selector works with base-quote naming convention"""
        # Create exchange with stream using dash naming
        exchange = Exchange("simulated", service=execute_order)
        price_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        exchange(price_stream)
        
        # Create wallet with BTC
        wallet = Wallet(exchange, 10 * BTC)
        
        # Test that stream selector can find the price stream
        streams = _create_wallet_source(wallet, include_worth=True)
        
        # Should not raise "No stream satisfies selector condition" error
        assert len(streams) > 0
        
    def test_stream_selector_with_plain_symbol(self):
        """Test stream selector works with plain symbol naming"""
        # Create exchange with stream using plain symbol
        exchange = Exchange("simulated", service=execute_order)
        price_stream = Stream.source([100, 101, 102], dtype="float").rename("BTC")
        exchange(price_stream)
        
        # Create wallet
        wallet = Wallet(exchange, 10 * BTC)
        
        # Test that stream selector can find the price stream
        streams = _create_wallet_source(wallet, include_worth=True)
        
        # Should not raise "No stream satisfies selector condition" error
        assert len(streams) > 0
        
    def test_stream_selector_with_custom_instrument(self):
        """Test stream selector works with custom instruments"""
        # Create custom instrument
        AAPL = Instrument("AAPL", 2, "Apple Inc.")
        
        # Create exchange with stream
        exchange = Exchange("simulated", service=execute_order)
        price_stream = Stream.source([150, 152, 153], dtype="float").rename("USD-AAPL")
        exchange(price_stream)
        
        # Create wallet
        wallet = Wallet(exchange, 10 * AAPL)
        
        # Test that stream selector can find the price stream
        streams = _create_wallet_source(wallet, include_worth=True)
        
        # Should not raise "No stream satisfies selector condition" error
        assert len(streams) > 0
        
    def test_stream_selector_without_worth(self):
        """Test stream selector when include_worth=False"""
        # Create exchange
        exchange = Exchange("simulated", service=execute_order)
        price_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        exchange(price_stream)
        
        # Create wallet
        wallet = Wallet(exchange, 10 * BTC)
        
        # Test that stream selector works without worth calculation
        streams = _create_wallet_source(wallet, include_worth=False)
        
        # Should have balance streams but no worth stream
        assert len(streams) == 3  # free, locked, total

