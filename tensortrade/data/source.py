
import pandas as pd

from abc import abstractmethod, ABCMeta
from typing import Dict, List


from tensortrade.base.core import TimeIndexed


class DataSource(TimeIndexed, metaclass=ABCMeta):

    def __init__(self):
        self._listeners = []

    def attach(self, listener):
        self._listeners += [listener]

    def detach(self, listener):
        self._listeners.remove(listener)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @abstractmethod
    def next(self) -> Dict[str, any]:
        raise NotImplementedError

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError()

    def reset(self):
        pass


class Array(DataSource):

    def __init__(self, array: List[any]):
        super().__init__()
        self._cursor = 0
        self._array = array

    def next(self) -> Dict:
        data = {self._cursor: self._array[self._cursor]}
        self._cursor += 1
        return data

    def has_next(self) -> bool:
        if self._cursor < len(self._array):
            return True
        return False

    def reset(self):
        self._cursor = 0


class DataFrame(DataSource):

    def __init__(self, data_frame: pd.DataFrame):
        super().__init__()
        self._cursor = 0
        self._data_frame = data_frame

    def next(self) -> Dict[str, any]:
        idx = self._data_frame.index[self._cursor]
        data = dict(self._data_frame.loc[idx, :])
        self._cursor += 1

        for listener in self._listeners:
            listener.on_next(data)

        return data

    def has_next(self) -> bool:
        if self._cursor < len(self._data_frame):
            return True
        return False

    def reset(self):
        self._cursor = 0



