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

import random
import numpy as np
import operator

from tensortrade.slippage import SlippageModel
from tensortrade.trades import Trade, TradeType


class RandomUniformSlippageModel(SlippageModel):
    """A uniform random slippage model."""

    def __init__(self, min: float = 0, max: float = 1, slip: float = 0) -> float:
        super().__init__()

        self.min = min
        self.max = max
        self.slip = slip
        pass

    def random_slip(self, number: float = 0) -> float:
        n = random.choice([self.slip_up(number),
                            self.slip_down(number)])
        return n

    def slip_up(self, number: float = 0) -> float:
        slip = np.random.uniform(0, (self.slip / 100))
        n = (number + (number*slip))
        n = min(self.max,  max(self.min, n))
        return n

    def slip_down(self, number: float = 0) -> float:
        slip = np.random.uniform(0, (self.slip/100))
        n = (number - (number*slip))
        n = min(self.max, max(self.min, n))
        return n
