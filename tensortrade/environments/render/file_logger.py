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


import os
from datetime import datetime
import logging
import pandas as pd

from tensortrade.environments.render import BaseRenderer


DEFAULT_FORMAT = '[%(asctime)-15s] %(message)s'
DEFAULT_DATEFMT = '%Y-%m-%d %H:%M:%S'


class FileLogger(BaseRenderer):
    def __init__(self, filename: str = None, path: str = 'log', format=None, datefmt: str = None, error: str = 'create'):
        if path and not os.path.exists(path):
            if error == 'create':
                os.mkdir(path)
            elif error == 'raise':
                raise OSError(f"Path '{path}' not found.")
            else:
                raise ValueError(f"Acceptable 'error' values are 'create' or 'raise'. Found '{error}'.")

        if not filename:
            filename = 'log_{}.log'.format(datetime.now().strftime('%Y%m%d_%H_%M_%S'))

        self._logger = logging.getLogger(self.id)
        self._logger.setLevel(logging.INFO)

        pathname = os.path.join(path, filename) if path else filename
        handler = logging.FileHandler(pathname)
        handler.setFormatter(logging.Formatter(
            format if format is not None else DEFAULT_FORMAT,
            datefmt=datefmt if datefmt is not None else DEFAULT_DATEFMT
            ))
        self._logger.addHandler(handler)

    @property
    def can_save(self) -> bool:
        return False

    @property
    def can_reset(self) -> bool:
        return False

    @property
    def log_file(self) -> str:
        return self._logger.handlers[0].baseFilename

    def render(self, episode: int, max_episodes: int, step: int, max_steps: int,
               price_history: pd.DataFrame, net_worth: pd.Series,
               performance: pd.DataFrame, trades
               ):
        self._logger.info('Episode: {}/{} - Step: {}/{} - Performance:\n{}'.format(
            episode + 1,
            max_episodes,
            step,
            max_steps,
            str(performance)
        ))
