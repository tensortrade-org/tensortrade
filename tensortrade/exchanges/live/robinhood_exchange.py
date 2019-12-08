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

import numpy as np
import pandas as pd

from typing import Dict, List
from gym.spaces import Space, Box

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import Exchange


class RobinhoodExchange(Exchange):
    """An exchange for trading using the Robinhood API."""

    def __init__(self,  **kwargs):
        raise NotImplementedError()
