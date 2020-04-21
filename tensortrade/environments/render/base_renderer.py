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
from datetime import datetime
import pandas as pd

from tensortrade.base import Identifiable


class BaseRenderer(Identifiable, metaclass=ABCMeta):

    def __init__(self):
        self._max_episodes = None
        self._max_steps = None

    def _create_log_entry(self, episode: int = None, max_episodes: int = None,
                         step: int = None, max_steps: int = None,
                         date_format: str='%Y-%m-%d %H:%M:%S %p'):
        log_entry = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S %p")}]'

        if episode is not None:
            log_entry += f' Episode: {episode + 1}' + (f'/{max_episodes}' if max_episodes else '')

        if step is not None:
            log_entry += f' Step: {step}' + (f'/{max_steps}' if max_steps else '')

        return log_entry

    @abstractmethod
    def render(self, episode: int = None, max_episodes: int = None,
               step: int = None, max_steps: int = None,
               price_history: pd.DataFrame = None, net_worth: pd.Series = None,
               performance: pd.DataFrame = None, trades: 'OrderedDict' = None
               ):
        raise NotImplementedError()

    def save(self):
        pass

    def reset(self):
        pass
