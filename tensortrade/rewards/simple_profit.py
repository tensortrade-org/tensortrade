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

import pandas as pd
import numpy as np

from tensortrade.rewards import RewardScheme


class SimpleProfit(RewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size: int = 1):
        self.window_size = window_size

    def reset(self):
        pass

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a sliding window.

        Args:
            portfolio: The portfolio being used by the environment.

        Returns:
            The incremental increase in net worth over the previous `window_size` timesteps.
        """
        returns = portfolio.performance['net_worth'].diff()
        return sum(returns[-self.window_size:])
