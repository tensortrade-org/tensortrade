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


from datetime import datetime
import pandas as pd

from tensortrade.environments.render import AbstractRenderer


DEFAULT_FORMAT = '[%(asctime)-15s] %(message)s'


class ScreenLogger(AbstractRenderer):
    def __init__(self, datefmt: str = '%Y-%m-%d %H:%M:%S'):
        self.format = datefmt

    @property
    def can_save(self) -> bool:
        return False

    @property
    def can_reset(self) -> bool:
        return False

    def render(self, episode: int, max_episodes: int, step: int, max_steps: int,
               price_history: pd.DataFrame, net_worth: pd.Series,
               performance: pd.DataFrame, trades
               ):
        print('[{}] Episode: {}/{} - Step: {}/{}'.format(
            datetime.now().strftime(self.format),
            episode + 1,
            max_episodes,
            step,
            max_steps
        ))
