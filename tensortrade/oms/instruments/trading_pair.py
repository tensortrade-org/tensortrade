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

from typing import Any

from tensortrade.core.exceptions import InvalidTradingPair


class TradingPair:
    """A pair of financial instruments to be traded on an exchange.

    Parameters
    ----------
    base : `Instrument`
        The base instrument of the trading pair.
    quote : `Instrument`
        The quote instrument of the trading pair.

    Raises
    ------
    InvalidTradingPair
        Raises if base and quote instrument are equal.
    """

    def __init__(self, base: "Instrument", quote: "Instrument") -> None:
        if base == quote:
            raise InvalidTradingPair(base, quote)
        self.base = base
        self.quote = quote

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: "Any") -> bool:
        if isinstance(other, TradingPair):
            if str(self) == str(other):
                return True
        return False

    def __ne__(self, other: "Any") -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return "{}/{}".format(self.base.symbol, self.quote.symbol)

    def __repr__(self) -> str:
        return str(self)
