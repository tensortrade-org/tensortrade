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


import pandas as pd

from abc import abstractmethod
from typing import Dict, List, Callable


from tensortrade.base.core import TimeIndexed
from tensortrade.data.stream.node import Node


class DataSource(TimeIndexed, Node):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def generate(self, inbound_data: dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class ArraySource(DataSource):

    def __init__(self, name: str, array: List[any] = None):
        super().__init__(name)

        self._array = array if array else []
        self._cursor = 0

    def generate(self) -> any:
        v = self._array[self._cursor]

        self._cursor += 1

        return v

    def has_next(self) -> bool:
        if self._cursor < len(self._array):
            return True
        return False

    def reset(self):
        self._cursor = 0


class DataFrameSource(DataSource):

    def __init__(self, name: str, data_frame: pd.DataFrame):
        super().__init__(name)

        self._data_frame = data_frame
        self._cursor = 0

    def generate(self) -> Dict[str, any]:
        idx = self._data_frame.index[self._cursor]
        data = dict(self._data_frame.loc[idx, :])

        self._cursor += 1

        return data

    def has_next(self) -> bool:
        if self._cursor < len(self._data_frame):
            return True
        return False

    def reset(self):
        self._cursor = 0


class LambdaSource(DataSource):

    def __init__(self, name: str, extract: Callable[[any], float], obj: any):
        super().__init__(name)

        self.extract = extract
        self.obj = obj

    def generate(self):
        return self.extract(self.obj)

    def has_next(self):
        return True

    def reset(self):
        pass
