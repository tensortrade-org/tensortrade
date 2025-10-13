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

from tensortrade.feed.core import Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order


class TestExchangeStreams:
    """Test suite for exchange stream naming consistency"""

    def test_exchange_stream_naming_convention(self):
        """Test that exchange streams follow consistent naming convention"""
        exchange = Exchange("bitfinex", service=execute_order)
        price_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        exchange(price_stream)
        
        # Get streams from exchange
        streams = exchange.streams()
        
        # Verify stream naming follows exchange_name:/base-quote format
        assert len(streams) == 1
        assert streams[0].name == "bitfinex:/USD-BTC"
        
    def test_multiple_exchange_streams(self):
        """Test multiple streams on single exchange"""
        exchange = Exchange("bitfinex", service=execute_order)
        btc_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        eth_stream = Stream.source([2000, 2010, 2020], dtype="float").rename("USD-ETH")
        exchange(btc_stream, eth_stream)
        
        # Get streams from exchange
        streams = exchange.streams()
        
        # Verify both streams are properly named
        assert len(streams) == 2
        stream_names = [s.name for s in streams]
        assert "bitfinex:/USD-BTC" in stream_names
        assert "bitfinex:/USD-ETH" in stream_names
        
    def test_exchange_stream_with_special_characters(self):
        """Test stream naming with special characters"""
        exchange = Exchange("test-exchange", service=execute_order)
        price_stream = Stream.source([100, 101, 102], dtype="float").rename("USD/BTC")
        exchange(price_stream)
        
        # Get streams from exchange
        streams = exchange.streams()
        
        # Verify stream naming handles special characters
        assert len(streams) == 1
        assert "test-exchange:/" in streams[0].name
        
    def test_exchange_is_pair_tradable(self):
        """Test that exchange correctly identifies tradable pairs"""
        exchange = Exchange("bitfinex", service=execute_order)
        btc_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        exchange(btc_stream)
        
        # Create trading pair
        from tensortrade.oms.instruments import USD, BTC
        pair = USD / BTC
        
        # Verify pair is tradable
        assert exchange.is_pair_tradable(pair)
        
    def test_exchange_quote_price(self):
        """Test that exchange returns correct quote price"""
        exchange = Exchange("bitfinex", service=execute_order)
        btc_stream = Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
        exchange(btc_stream)
        
        # Run the stream once to get a value
        btc_stream.run()
        
        # Create trading pair
        from tensortrade.oms.instruments import USD, BTC
        pair = USD / BTC
        
        # Get quote price
        price = exchange.quote_price(pair)
        
        # Verify price is correct
        assert price > 0

