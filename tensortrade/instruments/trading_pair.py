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


from numbers import Number

from tensortrade.base.exceptions import InvalidTradingPair, IncompatibleTradingPairOperation
from tensortrade.instruments.quantity import Price


class TradingPair:
    """A pair of financial instruments to be traded on a specific exchange."""

    def __init__(self, base: 'Instrument', quote: 'Instrument'):

        if base == quote:
            raise InvalidTradingPair(base, quote)

        self._base = base
        self._quote = quote

    @property
    def base(self):
        return self._base

    @property
    def quote(self):
        return self._quote

    def __rmul__(self, other):
        if not isinstance(other, Number):
            raise IncompatibleTradingPairOperation(other, self)
        return Price(other, self)

    def __eq__(self, other):
        if isinstance(other, TradingPair):
            if str(self) == str(other):
                return True
        return False

    def __str__(self):
        return '{}/{}'.format(self.base.symbol, self.quote.symbol)

    def __repr__(self):
        return str(self)
