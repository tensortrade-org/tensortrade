# Copyright 2019 The TensorTrade Authors.
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

from enum import Enum


class TradeType(Enum):
    """A trade type for use within trading environments."""

    HOLD = "hold"
    LIMIT_BUY = "limit_buy"
    MARKET_BUY = "market_buy"
    LIMIT_SELL = "limit_sell"
    MARKET_SELL = "market_sell"

    def is_buy(self) -> bool:
        return self == TradeType.MARKET_BUY or self == TradeType.LIMIT_BUY

    def is_sell(self) -> bool:
        return self == TradeType.MARKET_SELL or self == TradeType.LIMIT_SELL
