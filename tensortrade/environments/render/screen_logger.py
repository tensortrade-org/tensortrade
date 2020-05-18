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

from tensortrade.environments.render import BaseRenderer

DEFAULT_FORMAT = '[%(asctime)-15s] %(message)s'


class ScreenLogger(BaseRenderer):
    def __init__(self, date_format: str = '%Y-%m-%d %H:%M:%S %p'):
        self._date_format = date_format

    def render(self, episode: int = None, max_episodes: int = None,
               step: int = None, max_steps: int = None,
               price_history: pd.DataFrame = None, net_worth: pd.Series = None,
               performance: pd.DataFrame = None, trades: 'OrderedDict' = None
               ):
        print(self._create_log_entry(episode, max_episodes, step, max_steps, date_format=self._date_format))
