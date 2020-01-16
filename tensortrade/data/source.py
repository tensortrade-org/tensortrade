
import pandas as pd

from abc import abstractmethod, ABCMeta
from typing import Dict, List


from tensortrade.base.core import TimeIndexed


class DataSource(TimeIndexed, metaclass=ABCMeta):

    count = 0

    def __init__(self, name: str = None):
        self._outbound_sources = []
        self._name = name if name else str(DataSource.count)
        self._incoming_data = {}

        DataSource.count += 1

    @property
    def name(self):
        return self._name

    def add(self, source: 'DataSource'):
        self._outbound_sources += [source]

    def use(self, sources: List['DataSource']):
        for s in sources:
            s.add(self)

    def send(self, data: dict):
        for source in self._outbound_sources:
            source.receive(data)

    def receive(self, data: dict):
        self._incoming_data.update(data)

    def next(self, ) -> Dict[str, any]:
        data = self.call(self._incoming_data)
        self.send(data)
        self._incoming_data = {}
        return data

    @abstractmethod
    def call(self, data: dict):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> bool:
        raise NotImplementedError()

    def refresh(self):
        self.reset()
        for source in self._outbound_sources:
            source.refresh()


class Array(DataSource):

    def __init__(self, name: str, array: List[any] = None):
        super().__init__(name)
        self._cursor = 0
        self._array = array if array else []

    def call(self, data: dict) -> Dict:
        data = {self.name: self._array[self._cursor]}
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

    def call(self, data: dict) -> Dict[str, any]:
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
