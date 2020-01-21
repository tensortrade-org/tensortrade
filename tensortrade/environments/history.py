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


class History(object):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.observations = pd.DataFrame()

    def _flatten(self, observation: dict, parent_key='', sep=':/') -> dict:
        items = []

        for key, val in observation.items():
            new_key = parent_key + sep + key if parent_key else key

            if isinstance(val, collections.MutableMapping):
                items.extend(self._flatten(val, new_key, sep=sep).items())
            else:
                items.append((new_key, val))

        return dict(items)

    def push(self, observation: dict):
        """Saves an observation."""
        flattened = self._flatten(observation)

        self.observations = self.observations.append(flattened, ignore_index=True)

        if len(self.observations) > self.window_size:
            self.observations = self.observations[-self.window_size:]

    def observe(self) -> np.array:
        """Returns the observation to be observed by the agent."""
        observation = self.observations.copy()

        if len(observation) < self.window_size:
            size = self.window_size - len(observation)
            padding = np.zeros((size, observation.shape[1]))
            padding = pd.DataFrame(padding, columns=self.observations.columns)
            observation = pd.concat([padding, observation], ignore_index=True, sort=False)

        if isinstance(observation, pd.DataFrame):
            observation = observation.fillna(0, axis=1)
            observation = observation.values

        observation = np.nan_to_num(observation)

        return observation

    def reset(self):
        self.observations = pd.DataFrame()
