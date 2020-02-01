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
# limitations under the License.


import collections
import pandas as pd
import numpy as np


class ObservationHistory(object):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.rows = pd.DataFrame()

    def push(self, row: dict):
        """Saves an observation."""
        self.rows = self.rows.append(row, ignore_index=True)

        if len(self.rows) > self.window_size:
            self.rows = self.rows[-self.window_size:]

    def observe(self) -> np.array:
        """Returns the rows to be observed by the agent."""
        rows = self.rows.copy()

        if len(rows) < self.window_size:
            size = self.window_size - len(rows)
            padding = np.zeros((size, rows.shape[1]))
            padding = pd.DataFrame(padding, columns=self.rows.columns)
            rows = pd.concat([padding, rows], ignore_index=True, sort=False)

        if isinstance(rows, pd.DataFrame):
            rows = rows.fillna(0, axis=1)
            rows = rows.values

        rows = np.nan_to_num(rows)

        return rows

    def reset(self):
        self.rows = pd.DataFrame()
