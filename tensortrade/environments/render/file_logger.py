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
import logging
import pandas as pd

from tensortrade.environments.render import BaseRenderer
from tensortrade.environments.utils.helpers import create_auto_file_name, check_path

DEFAULT_LOG_FORMAT = '[%(asctime)-15s] %(message)s'
DEFAULT_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'


class FileLogger(BaseRenderer):
    def __init__(self, filename: str = None, path: str = 'log', log_format=None,
                 timestamp_format: str = None):
        """
        Arguments:
            filename: The file name of the log file. If omitted, a file name will be
                created automatically.
            path: The path to save the log files to. None to save to same script directory.
            log_format: The log entry format as per Python logging. None for default. For
                more details, refer to https://docs.python.org/3/library/logging.html
            timestamp_format: The format of the timestamp of the log entry. Node for default.
        """
        check_path(path)

        if not filename:
            filename = create_auto_file_name('log_', 'log')

        self._logger = logging.getLogger(self.id)
        self._logger.setLevel(logging.INFO)

        if path:
            filename = os.path.join(path, filename)
        handler = logging.FileHandler(filename)
        handler.setFormatter(logging.Formatter(
            log_format if log_format is not None else DEFAULT_LOG_FORMAT,
            datefmt=timestamp_format if timestamp_format is not None else DEFAULT_TIMESTAMP_FORMAT
            ))
        self._logger.addHandler(handler)

    @property
    def log_file(self) -> str:
        return self._logger.handlers[0].baseFilename

    def render(self, episode: int = None, max_episodes: int = None,
               step: int = None, max_steps: int = None,
               price_history: pd.DataFrame = None, net_worth: pd.Series = None,
               performance: pd.DataFrame = None, trades: 'OrderedDict' = None
               ):
        log_entry = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._logger.info('{} - Performance:\n{}'.format(log_entry, str(performance)))
