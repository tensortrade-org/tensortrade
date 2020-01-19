
import pandas as pd

from abc import abstractmethod
from typing import Dict, List, Callable


from tensortrade.base.core import TimeIndexed
from tensortrade.data.stream.node import Node


class DataSource(TimeIndexed, Node):

    def __init__(self, name):
        super().__init__(name)

    def call(self, inbound_data: dict):
        return self.generate()

    @abstractmethod
    def generate(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class Array(DataSource):

    def __init__(self, name: str, array: List[any] = None):
        super().__init__(name)
        self._cursor = 0
        self._array = array if array else []

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


class DataFrame(DataSource):

    def __init__(self, name: str, data_frame: pd.DataFrame):
        super().__init__(name)
        self._cursor = 0
        self._data_frame = data_frame

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


class Lambda(DataSource):

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