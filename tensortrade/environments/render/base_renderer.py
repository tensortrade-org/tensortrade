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


from abc import ABCMeta, abstractmethod
from tensortrade.base import Identifiable
import pandas as pd


class BaseRenderer(Identifiable, metaclass=ABCMeta):

    def __init__(self):
        self._max_episodes = None
        self._max_steps = None

    @abstractmethod
    def render(self, episode: int, max_episodes: int, step: int, max_steps: int,
               price_history: pd.DataFrame, net_worth: pd.Series,
               performance: pd.DataFrame, trades
               ):
        raise NotImplementedError()

    def save(self):
        pass

    def reset(self):
        pass
